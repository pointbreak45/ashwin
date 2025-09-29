"""
FastAPI Backend for Indian Education Law Chatbot
Provides REST API endpoints for legal document search and chat functionality
"""

import os
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path

# FastAPI imports
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Import our services
import sys
sys.path.append(str(Path(__file__).parent.parent))

try:
    from services.vector_service import VectorEmbeddingService
    from utils.data_loader import LegalDocumentLoader
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure to install all dependencies: pip install -r requirements.txt")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Indian Education Law Chatbot API",
    description="API for searching and querying Indian education law documents",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],  # React frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for services
vector_service: Optional[VectorEmbeddingService] = None
data_loader: Optional[LegalDocumentLoader] = None

# Pydantic models for request/response
class ChatRequest(BaseModel):
    question: str = Field(..., description="User's legal question", min_length=1, max_length=1000)
    context: str = Field("", description="Additional context for the question", max_length=2000)

class ChatResponse(BaseModel):
    answer: str = Field(..., description="Generated answer to the legal question")
    sources: List[Dict[str, Any]] = Field(default=[], description="Relevant legal document sources")
    confidence: float = Field(..., description="Confidence score for the answer")
    query_time: float = Field(..., description="Time taken to process the query in seconds")
    disclaimer: str = Field(..., description="Legal disclaimer")

class DocumentSearchRequest(BaseModel):
    query: str = Field(..., description="Search query", min_length=1, max_length=500)
    limit: int = Field(5, description="Number of results to return", ge=1, le=20)
    score_threshold: float = Field(0.0, description="Minimum similarity score", ge=0.0, le=1.0)

class DocumentSearchResponse(BaseModel):
    results: List[Dict[str, Any]] = Field(..., description="Search results")
    total_found: int = Field(..., description="Total number of results found")
    query_time: float = Field(..., description="Time taken to search in seconds")

class SystemHealthResponse(BaseModel):
    status: str = Field(..., description="System status")
    vector_index_loaded: bool = Field(..., description="Whether vector index is loaded")
    total_documents: int = Field(..., description="Total number of indexed documents")
    model_name: str = Field(..., description="Name of the embedding model")
    uptime: str = Field(..., description="System uptime")

# Initialize services
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    global vector_service, data_loader
    
    try:
        logger.info("Starting up Indian Education Law Chatbot API...")
        
        # Initialize data loader
        data_loader = LegalDocumentLoader()
        
        # Initialize vector service
        vector_service = VectorEmbeddingService()
        
        # Try to load existing vector index
        try:
            vector_service.load_index_for_search()
            logger.info("Loaded existing vector index")
        except FileNotFoundError:
            logger.info("No existing vector index found. Building new index...")
            await build_vector_index()
        
        logger.info("API startup completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")
        raise

async def build_vector_index():
    """Build vector index from available documents"""
    global vector_service, data_loader
    
    try:
        # Load processed documents or create them
        processed_docs = data_loader.load_processed_documents()
        
        if not processed_docs:
            logger.info("No processed documents found. Processing raw documents...")
            raw_docs = data_loader.load_json_documents()
            if raw_docs:
                processed_docs = data_loader.process_documents(raw_docs)
                data_loader.save_processed_documents(processed_docs)
            else:
                logger.warning("No documents found to index!")
                return
        
        # Build vector index
        if processed_docs:
            logger.info(f"Building vector index for {len(processed_docs)} documents...")
            vector_service.build_and_save_index(processed_docs)
            logger.info("Vector index built successfully!")
        
    except Exception as e:
        logger.error(f"Error building vector index: {str(e)}")
        raise

# API Routes
@app.get("/", response_class=JSONResponse)
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Indian Education Law Chatbot API",
        "version": "1.0.0",
        "status": "active",
        "documentation": "/docs"
    }

@app.get("/api/health", response_model=SystemHealthResponse)
async def get_system_health():
    """Get system health status"""
    try:
        if not vector_service:
            raise HTTPException(status_code=500, detail="Vector service not initialized")
        
        stats = vector_service.get_index_statistics()
        
        return SystemHealthResponse(
            status="healthy" if stats.get("total_vectors", 0) > 0 else "no_data",
            vector_index_loaded=stats.get("total_vectors", 0) > 0,
            total_documents=stats.get("total_vectors", 0),
            model_name=stats.get("model_name", "unknown"),
            uptime="active"
        )
        
    except Exception as e:
        logger.error(f"Error getting system health: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/search", response_model=DocumentSearchResponse)
async def search_documents(request: DocumentSearchRequest):
    """Search for relevant legal documents"""
    import time
    start_time = time.time()
    
    try:
        if not vector_service:
            raise HTTPException(status_code=500, detail="Vector service not initialized")
        
        # Perform vector search
        results = vector_service.search_similar_documents(
            query=request.query,
            k=request.limit,
            score_threshold=request.score_threshold
        )
        
        query_time = time.time() - start_time
        
        return DocumentSearchResponse(
            results=results,
            total_found=len(results),
            query_time=round(query_time, 3)
        )
        
    except Exception as e:
        logger.error(f"Error searching documents: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat", response_model=ChatResponse)
async def chat_with_legal_assistant(request: ChatRequest):
    """Main chat endpoint for legal questions"""
    import time
    start_time = time.time()
    
    try:
        if not vector_service:
            raise HTTPException(status_code=500, detail="Vector service not initialized")
        
        # Search for relevant documents
        search_results = vector_service.search_similar_documents(
            query=request.question,
            k=3,  # Get top 3 most relevant documents
            score_threshold=0.1
        )
        
        # Generate answer based on search results
        answer = generate_legal_answer(request.question, search_results, request.context)
        
        # Calculate confidence based on search scores
        confidence = calculate_confidence(search_results)
        
        query_time = time.time() - start_time
        
        # Prepare sources
        sources = []
        for result in search_results:
            sources.append({
                "section": result["document"]["section"],
                "title": result["document"]["title"],
                "year": result["document"]["year"],
                "score": result["score"],
                "content_preview": result["document"]["content"][:200] + "..." if len(result["document"]["content"]) > 200 else result["document"]["content"]
            })
        
        return ChatResponse(
            answer=answer,
            sources=sources,
            confidence=confidence,
            query_time=round(query_time, 3),
            disclaimer="This system provides information from Indian education law sources. It is not a substitute for professional legal advice."
        )
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def generate_legal_answer(question: str, search_results: List[Dict[str, Any]], context: str = "") -> str:
    """
    Generate a legal answer based on search results
    
    This is a simple implementation. In production, you might want to use
    a more sophisticated language model like GPT or a fine-tuned legal LLM.
    """
    if not search_results:
        return """No clear answer found in the uploaded documents. Please consult official sources or a lawyer.

For Indian education law matters, I recommend:
1. Consulting the official Ministry of Education website
2. Reviewing the Right to Education Act, 2009
3. Checking UGC regulations for higher education
4. Seeking advice from qualified legal professionals

This system provides information from Indian education law sources. It is not a substitute for professional legal advice."""

    # Get the most relevant document
    top_result = search_results[0]
    top_doc = top_result["document"]
    
    # Create a structured answer
    answer_parts = []
    
    # Add relevant legal provision
    answer_parts.append(f"Based on the legal documents in our database, here's what I found regarding your question:\n")
    
    # Add the most relevant section
    if top_result["score"] > 0.3:  # High confidence
        answer_parts.append(f"**{top_doc['section']} — {top_doc['title']} — {top_doc['year']}**")
        answer_parts.append(f"\n{top_doc['content']}\n")
    else:
        answer_parts.append(f"The most relevant provision found is **{top_doc['section']}** from {top_doc['title']} ({top_doc['year']}), though it may only partially address your question.")
    
    # Add additional context if multiple results
    if len(search_results) > 1:
        answer_parts.append(f"\n**Related Provisions:**")
        for result in search_results[1:3]:  # Show up to 2 additional results
            doc = result["document"]
            answer_parts.append(f"• {doc['section']} — {doc['title']} — {doc['year']}")
    
    # Add disclaimer and guidance
    answer_parts.append(f"\n**Important:** This information is based on the documents in our legal database. Always verify with official sources and consult qualified legal professionals for specific legal advice.")
    
    return "\n".join(answer_parts)

def calculate_confidence(search_results: List[Dict[str, Any]]) -> float:
    """Calculate confidence score based on search results"""
    if not search_results:
        return 0.0
    
    # Use the score of the top result as base confidence
    top_score = search_results[0]["score"]
    
    # Adjust confidence based on number of good matches
    good_matches = sum(1 for result in search_results if result["score"] > 0.2)
    
    # Confidence calculation
    confidence = min(top_score * (1 + good_matches * 0.1), 1.0)
    
    return round(confidence, 3)

@app.post("/api/rebuild-index")
async def rebuild_vector_index(background_tasks: BackgroundTasks):
    """Rebuild the vector index (admin endpoint)"""
    try:
        background_tasks.add_task(build_vector_index)
        return {"message": "Vector index rebuild started in background"}
        
    except Exception as e:
        logger.error(f"Error starting index rebuild: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/stats")
async def get_system_statistics():
    """Get detailed system statistics"""
    try:
        if not vector_service or not data_loader:
            raise HTTPException(status_code=500, detail="Services not initialized")
        
        # Get vector index stats
        vector_stats = vector_service.get_index_statistics()
        
        # Get document stats
        processed_docs = data_loader.load_processed_documents()
        doc_stats = data_loader.get_document_statistics(processed_docs)
        
        return {
            "vector_index": vector_stats,
            "documents": doc_stats,
            "api_status": "active",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting system statistics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"detail": "Endpoint not found", "message": "Please check the API documentation at /docs"}
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    logger.error(f"Internal server error: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "message": "Please check the server logs"}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )