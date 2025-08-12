import os
import sys
import json
import logging
from typing import Annotated, Optional
from pathlib import Path

from fastapi import FastAPI, Header, Query, HTTPException, Request
from prometheus_fastapi_instrumentator import Instrumentator
import httpx

sys.path.insert(0, os.path.abspath(".."))

from common.scheme import (
    ContentAnalysisHttpRequest,
    ContentsAnalysisResponse,
    ContentAnalysisResponse,
    Error,
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
KSERVE_INTERNAL_URL = os.getenv("KSERVE_INTERNAL_URL")  # Internal cluster URL
KSERVE_EXTERNAL_URL = os.getenv("KSERVE_EXTERNAL_URL")  # External route URL
PREFER_INTERNAL = os.getenv("PREFER_INTERNAL", "true").lower() == "true"
TIMEOUT_SECONDS = int(os.getenv("TIMEOUT_SECONDS", "30"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))

# SSL Configuration
SSL_VERIFY = os.getenv("SSL_VERIFY", "true").lower() == "true"
CA_CERT_PATH = os.getenv("CA_CERT_PATH", "/var/run/secrets/kubernetes.io/serviceaccount/ca.crt")

def _parse_safe_labels_env():
    """Parse safe labels from environment variable (same logic as detector.py)"""
    if os.environ.get("SAFE_LABELS"):
        try:
            parsed = json.loads(os.environ.get("SAFE_LABELS"))
            if isinstance(parsed, (int, str)):
                logger.info(f"SAFE_LABELS env var: {parsed}")
                return [parsed]
            if isinstance(parsed, list) and all(isinstance(x, (int, str)) for x in parsed):
                logger.info(f"SAFE_LABELS env var: {parsed}")
                return parsed
        except Exception as e:
            logger.warning(f"Could not parse SAFE_LABELS env var: {e}. Defaulting to [0].")
            return [0]
    logger.info("SAFE_LABELS env var not set: defaulting to [0].")
    return [0]

SAFE_LABELS = _parse_safe_labels_env()

def get_service_account_token() -> Optional[str]:
    """Get ServiceAccount token for service-to-service authentication."""
    token_path = "/var/run/secrets/kubernetes.io/serviceaccount/token"
    try:
        if Path(token_path).exists():
            with open(token_path, 'r') as f:
                token = f.read().strip()
                logger.debug("ServiceAccount token loaded successfully")
                return token
    except Exception as e:
        logger.warning(f"Could not read ServiceAccount token: {e}")
    return None

def extract_user_token(request: Request) -> Optional[str]:
    """Extract user Bearer token from request headers."""
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header[7:] 
        logger.debug("User Bearer token extracted from request")
        return token
    return None

def get_ssl_config(url: str) -> dict:
    """Get SSL configuration based on URL type."""
    if url and ".svc.cluster.local" in url:
        # Internal cluster communication - explicitly disable SSL verification
        logger.info(f"Internal cluster URL detected: {url} - disabling SSL verification")
        return {"verify": False}
    else:
        # External communication
        logger.info(f"External URL detected: {url} - SSL verify: {SSL_VERIFY}")
        return {"verify": SSL_VERIFY}

def select_kserve_url_and_auth(user_token: Optional[str] = None) -> tuple[str, Optional[str]]:
    """
    Select appropriate KServe URL and authentication token.
    
    Returns:
        tuple: (url, auth_token)
    """
    service_account_token = get_service_account_token()
    
    # Priority 1: Internal URL with ServiceAccount token 
    if PREFER_INTERNAL and KSERVE_INTERNAL_URL and service_account_token:
        logger.info("Using internal URL with ServiceAccount token")
        return KSERVE_INTERNAL_URL, service_account_token
    
    # Priority 2: External URL with user token 
    if KSERVE_EXTERNAL_URL and user_token:
        logger.info("Using external URL with user token")
        return KSERVE_EXTERNAL_URL, user_token
    
    # Priority 3: Internal URL with user token
    if KSERVE_INTERNAL_URL and user_token:
        logger.info("Using internal URL with user token (fallback)")
        return KSERVE_INTERNAL_URL, user_token
    
    # Priority 4: External URL with ServiceAccount token
    if KSERVE_EXTERNAL_URL and service_account_token:
        logger.warning("Using external URL with ServiceAccount token (not recommended)")
        return KSERVE_EXTERNAL_URL, service_account_token
    
    # Fallback: Use any available URL without auth
    if KSERVE_INTERNAL_URL:
        logger.warning("Using internal URL without authentication")
        return KSERVE_INTERNAL_URL, None
    elif KSERVE_EXTERNAL_URL:
        logger.warning("Using external URL without authentication")
        return KSERVE_EXTERNAL_URL, None
    
    raise HTTPException(
        status_code=500,
        detail="No valid KServe URL configured. Set KSERVE_INTERNAL_URL or KSERVE_EXTERNAL_URL"
    )

# Validate configuration
if not KSERVE_INTERNAL_URL and not KSERVE_EXTERNAL_URL:
    logger.error("At least one KServe URL must be configured")
    raise ValueError("KSERVE_INTERNAL_URL or KSERVE_EXTERNAL_URL must be set")

logger.info(f"KServe Internal URL: {KSERVE_INTERNAL_URL}")
logger.info(f"KServe External URL: {KSERVE_EXTERNAL_URL}")
logger.info(f"Prefer Internal: {PREFER_INTERNAL}")
logger.info(f"Safe labels: {SAFE_LABELS}")

app = FastAPI(
    title="KServe HuggingFace Detector",
    description="Production-ready detector service with dual URL and auth support",
    version="1.0.0"
)

Instrumentator().instrument(app).expose(app)

async def call_kserve_model(texts: list, request: Request, override_url: str = None) -> list:
    """Call KServe model endpoint with automatic URL and auth selection."""
    
    if override_url:
        url = override_url
        user_token = extract_user_token(request)
        auth_token = user_token or get_service_account_token()
    else:
        user_token = extract_user_token(request)
        url, auth_token = select_kserve_url_and_auth(user_token)
    
    if not url:
        raise HTTPException(status_code=500, detail="No KServe URL available")
    
    # Prepare headers
    headers = {"Content-Type": "application/json"}
    if auth_token:
        headers["Authorization"] = f"Bearer {auth_token}"
    
    payload = {"instances": texts}
    ssl_config = get_ssl_config(url)  # This should return {"verify": False} for internal URLs
    timeout = httpx.Timeout(TIMEOUT_SECONDS)
    
    logger.info(f"Calling KServe: {url}")
    logger.info(f"SSL config: {ssl_config}") 
    logger.info(f"Auth token present: {'Yes' if auth_token else 'No'}")
    
    for attempt in range(MAX_RETRIES):
        try:
            # Explicitly pass the SSL config
            async with httpx.AsyncClient(timeout=timeout, **ssl_config) as client:
                logger.info(f"Attempt {attempt + 1}/{MAX_RETRIES}")
                response = await client.post(url, json=payload, headers=headers)
                response.raise_for_status()
                
                result = response.json()
                predictions = result.get("predictions", [])
                
                if not predictions:
                    raise HTTPException(status_code=502, detail="No predictions returned")
                
                logger.info(f"Successfully received {len(predictions)} predictions")
                return predictions
                
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP {e.response.status_code}: {e.response.text}")
            if attempt == MAX_RETRIES - 1:
                raise HTTPException(
                    status_code=502,
                    detail=f"KServe HTTP error: {e.response.status_code}"
                )
        except httpx.TimeoutException:
            logger.error(f"Timeout on attempt {attempt + 1}")
            if attempt == MAX_RETRIES - 1:
                raise HTTPException(status_code=504, detail="KServe timeout")
        except Exception as e:
            logger.error(f"Error on attempt {attempt + 1}: {str(e)}")
            if attempt == MAX_RETRIES - 1:
                raise HTTPException(status_code=502, detail=f"KServe error: {str(e)}")
    

def process_sequence_classification_predictions(predictions: list, texts: list, detector_params: dict = None) -> list:
    """Process KServe predictions with same logic as detector.py sequence classification."""
    detector_params = detector_params or {}
    threshold = detector_params.get("threshold", 0.5)
    
    request_safe_labels = set(detector_params.get("safe_labels", []))
    all_safe_labels = set(SAFE_LABELS) | request_safe_labels
    
    logger.info(f"Processing {len(predictions)} predictions with threshold={threshold}")
    
    contents_analyses = []
    
    for text, prediction in zip(texts, predictions):
        content_analyses = []
        if isinstance(prediction, dict):
            for label_str, prob in prediction.items():
                try:
                    idx = int(label_str)
                    if prob >= threshold and idx not in all_safe_labels:
                        logger.info(f"Detection: label_{idx} score={prob:.4f}")
                        content_analyses.append(
                            ContentAnalysisResponse(
                                start=0,
                                end=len(text),
                                detection="sequence_classification",
                                detection_type=f"label_{idx}",
                                score=float(prob),
                                text=text,
                                evidences=[],
                            )
                        )
                except (ValueError, TypeError) as e:
                    logger.warning(f"Invalid label '{label_str}': {e}")
        else:
            logger.warning(f"Unexpected prediction format: {type(prediction)}")
        
        contents_analyses.append(content_analyses)
    
    return contents_analyses

@app.post("/api/v1/text/contents", response_model=ContentsAnalysisResponse)
async def detector_unary_handler(
    request: ContentAnalysisHttpRequest,
    http_request: Request,
    detector_id: Annotated[str, Header(example="hap")],
    kserve_url: Optional[str] = Query(None, description="Override KServe URL"),
):
    """Process content analysis with automatic URL and auth selection."""
    if not request.contents:
        raise HTTPException(status_code=422, detail="No content provided")
    
    logger.info(f"Processing {len(request.contents)} texts for {detector_id}")
    
    try:
        predictions = await call_kserve_model(request.contents, http_request, kserve_url)
        
        result = process_sequence_classification_predictions(
            predictions,
            request.contents,
            request.detector_params or {} 
        )
        
        return ContentsAnalysisResponse(root=result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check with connectivity validation."""
    internal_ok = False
    external_ok = False
    
    # Test internal URL
    if KSERVE_INTERNAL_URL:
        try:
            sa_token = get_service_account_token()
            headers = {}
            if sa_token:
                headers["Authorization"] = f"Bearer {sa_token}"
            
            ssl_config = get_ssl_config(KSERVE_INTERNAL_URL)
            async with httpx.AsyncClient(timeout=httpx.Timeout(5.0), **ssl_config) as client:
                test_url = KSERVE_INTERNAL_URL.replace('/v1/models/hap:predict', '')
                response = await client.get(test_url, headers=headers)
                internal_ok = response.status_code < 500
        except Exception as e:
            logger.debug(f"Internal health check failed: {e}")
    
    # Test external URL
    if KSERVE_EXTERNAL_URL:
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(5.0)) as client:
                test_url = KSERVE_EXTERNAL_URL.replace('/v1/models/hap:predict', '')
                response = await client.get(test_url)
                external_ok = response.status_code < 500
        except Exception as e:
            logger.debug(f"External health check failed: {e}")
    
    status = "healthy" if (internal_ok or external_ok) else "degraded"
    
    return {
        "status": status,
        "internal_url": KSERVE_INTERNAL_URL,
        "external_url": KSERVE_EXTERNAL_URL,
        "internal_accessible": internal_ok,
        "external_accessible": external_ok,
        "prefer_internal": PREFER_INTERNAL,
        "service_account_available": get_service_account_token() is not None,
        "ssl_verify": SSL_VERIFY
    }

@app.get("/config")
async def get_config():
    """Get current configuration."""
    return {
        "internal_url": KSERVE_INTERNAL_URL,
        "external_url": KSERVE_EXTERNAL_URL,
        "prefer_internal": PREFER_INTERNAL,
        "safe_labels": SAFE_LABELS,
        "timeout_seconds": TIMEOUT_SECONDS,
        "max_retries": MAX_RETRIES,
        "ssl_verify": SSL_VERIFY
    }

@app.on_event("startup")
async def startup_event():
    logger.info("Starting KServe Detector with dual URL support")
    logger.info(f"ServiceAccount token: {'Available' if get_service_account_token() else 'Not available'}")