#!/usr/bin/env python3
"""
HTTP server for language detection and other simple endpoints
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from pipeline.compliant_pipeline import CompliantVideoTranslationPipeline

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Video Translation ML Service", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize pipeline
pipeline = CompliantVideoTranslationPipeline()

class LanguageDetectionRequest(BaseModel):
    file_path: str

class LanguageDetectionResponse(BaseModel):
    detected_language: str
    confidence: float
    success: bool
    message: str

class TranslationRequest(BaseModel):
    file_path: str
    source_lang: str
    target_lang: str

class TranslationResponse(BaseModel):
    success: bool
    message: str
    session_id: str
    output_path: str = None

@app.post("/detect-language", response_model=LanguageDetectionResponse)
async def detect_language(request: LanguageDetectionRequest) -> LanguageDetectionResponse:
    """Detect language from video file"""
    try:
        file_path = Path(request.file_path)
        
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found")
        
        logger.info(f"Detecting language for file: {file_path}")
        
        # Use the pipeline to detect language
        detected_language, confidence = await pipeline.detect_language(file_path)
        
        return LanguageDetectionResponse(
            detected_language=detected_language,
            confidence=confidence,
            success=True,
            message=f"Language detected: {detected_language} (confidence: {confidence:.2f})"
        )
        
    except Exception as e:
        logger.error(f"Language detection failed: {str(e)}")
        return LanguageDetectionResponse(
            detected_language="en",
            confidence=0.5,
            success=False,
            message=f"Language detection failed: {str(e)}"
        )

@app.post("/translate", response_model=TranslationResponse)
async def translate_video(request: TranslationRequest) -> TranslationResponse:
    """Translate video directly using Python pipeline"""
    try:
        import uuid
        session_id = str(uuid.uuid4())
        
        logger.info(f"Starting translation for session {session_id}")
        logger.info(f"File: {request.file_path}")
        logger.info(f"Source: {request.source_lang} -> Target: {request.target_lang}")
        
        # Create output path
        output_path = Path(f"outputs/{session_id}_translated.mp4")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Run the pipeline directly
        result = await pipeline.process_video(
            video_path=Path(request.file_path),
            source_lang=request.source_lang,
            target_lang=request.target_lang,
            output_path=output_path,
            session_id=session_id
        )
        
        if result.get('success', False):
            logger.info(f"Translation completed successfully for session {session_id}")
            return TranslationResponse(
                success=True,
                message="Translation completed successfully",
                session_id=session_id,
                output_path=result.get('output_path', '')
            )
        else:
            logger.error(f"Translation failed for session {session_id}: {result.get('error', 'Unknown error')}")
            return TranslationResponse(
                success=False,
                message=result.get('error', 'Translation failed'),
                session_id=session_id
            )
            
    except Exception as e:
        logger.error(f"Translation error: {e}", exc_info=True)
        return TranslationResponse(
            success=False,
            message=f"Translation failed: {str(e)}",
            session_id=""
        )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "ml-service"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=50052)

