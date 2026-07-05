from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import List
import uvicorn
import logging
from contextlib import asynccontextmanager
import os
import time

# Cap CPU threads BEFORE importing the model. Un-capped, onnxruntime/OpenMP spawn
# one thread per core, which slows single short-sequence requests on many-core
# hosts. Override via OMP_NUM_THREADS.
os.environ.setdefault("OMP_NUM_THREADS", "4")

from inference_onnx import AisleCategorizer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global model instance
categorizer = None

MODEL_PATH = './aisle_model_onnx'
MAX_ITEMS_PER_REQUEST = 50


# ============================================================
# REQUEST/RESPONSE MODELS
# ============================================================

class CategorizeRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "items": ["2 lbs chicken breast", "Fresh tomatoes", "Whole milk"],
            }
        }
    )
    
    items: List[str] = Field(
        ..., 
        min_length=1, 
        max_length=MAX_ITEMS_PER_REQUEST,
        description=f"List of grocery items to categorize (1-{MAX_ITEMS_PER_REQUEST} items)"
    )
    
    @field_validator('items')
    @classmethod
    def validate_items(cls, v):
        if not all(item.strip() for item in v):
            raise ValueError("Items cannot be empty strings")
        return [item.strip() for item in v]


class CategoryResult(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "item": "2 lbs chicken breast",
                "category": "meat",
                "confidence": 0.9842,
                "probabilities": None
            }
        }
    )
    
    item: str = Field(..., description="Original item text")
    category: str = Field(..., description="Predicted aisle (one of 14)")
    confidence: float = Field(..., description="Confidence score (0.0-1.0)")
    base: str = Field(..., description="Base food-type aisle from the base head")
    state: str = Field(..., description="Preservation state from the state head: none|frozen|canned")
    uncertain: bool = Field(
        default=False,
        description="Advisory only: True when calibrated confidence is below threshold. A category is always returned regardless."
    )


class CategorizeResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "results": [
                    {
                        "item": "2 lbs chicken breast",
                        "category": "meat",
                        "confidence": 0.9842,
                        "probabilities": None
                    }
                ],
                "processing_time_ms": 45.2,
            }
        }
    )
    
    results: List[CategoryResult] = Field(..., description="Categorization results")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")


class HealthResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "healthy",
            }
        }
    )
    
    status: str = Field(..., description="Service status")


# ============================================================
# LIFESPAN MANAGEMENT
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    global categorizer
    
    logger.info("Starting up...")
    logger.info(f"Loading model from {MODEL_PATH}")
    
    try:
        categorizer = AisleCategorizer(MODEL_PATH)
        logger.info("Model loaded successfully")
        from aisle_map import FINAL_AISLES
        logger.info(f"Aisles: {FINAL_AISLES}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise
    
    yield
    
    logger.info("Shutting down...")


# ============================================================
# CREATE APP
# ============================================================

app = FastAPI(
    title="Grocery Categorization API",
    description="Multilingual grocery item categorization service supporting multi-language",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# ENDPOINTS
# ============================================================

@app.get("/", status_code=status.HTTP_200_OK)
async def root():
    return {
        "service": "Grocery Categorization API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "categorize": "/categorize",
            "categories": "/categories",
        }
    }


@app.get("/health", response_model=HealthResponse, status_code=status.HTTP_200_OK)
async def health_check():
    if categorizer is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    return HealthResponse(
        status="healthy",
    )


@app.post("/categorize", response_model=CategorizeResponse, status_code=status.HTTP_200_OK)
async def categorize_items(request: CategorizeRequest):
    """
    Multilingual: item text may be in any language.

    Categories (14 aisles): produce, dairy, meat, seafood, bakery, baking,
    spices, grocery, condiments, beverages, liquor, nonfood, frozen, canned
    """
    if categorizer is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    try:
        start_time = time.time()
        
        predictions = categorizer.predict_batch(
            request.items,
        )
        
        results = []
        for item, pred in zip(request.items, predictions):
            results.append(CategoryResult(
                item=item,
                category=pred['category'],
                confidence=pred['confidence'],
                base=pred['base'],
                state=pred['state'],
                uncertain=pred.get('uncertain', False),
            ))
        
        processing_time_ms = (time.time() - start_time) * 1000
        
        logger.info(f"Categorized {len(request.items)} items in {processing_time_ms:.2f}ms")
        
        return CategorizeResponse(
            results=results,
            processing_time_ms=round(processing_time_ms, 2),
        )
        
    except Exception as e:
        logger.error(f"Error during categorization: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Categorization failed: {str(e)}"
        )


@app.get("/categories", status_code=status.HTTP_200_OK)
async def get_categories():
    if categorizer is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    from aisle_map import FINAL_AISLES
    return {
        "categories": FINAL_AISLES,
    }


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Grocery Categorization API")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    
    args = parser.parse_args()
    
    logger.info(f"Starting server on {args.host}:{args.port}")
    
    uvicorn.run(
        "server:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        reload=args.reload,
        log_level="info"
    )
