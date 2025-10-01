from contextlib import asynccontextmanager

from base_detector_registry import BaseDetectorRegistry
from fastapi import HTTPException
from file_type_detectors import FileTypeDetectorRegistry
from prometheus_fastapi_instrumentator import Instrumentator
from regex_detectors import RegexDetectorRegistry

from chunkers import get_chunker_registry
from detectors.common.app import DetectorBaseAPI as FastAPI
from detectors.common.scheme import (ContentAnalysisHttpRequest,
                                     ContentsAnalysisResponse)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.set_detector(RegexDetectorRegistry(), "regex")
    app.set_detector(FileTypeDetectorRegistry(), "file_type")
    yield
    app.cleanup_detector()


app = FastAPI(lifespan=lifespan)
Instrumentator().instrument(app).expose(app)
chunker_registry = get_chunker_registry()


@app.post("/api/v1/text/contents", response_model=ContentsAnalysisResponse)
def detect_content(request: ContentAnalysisHttpRequest):
    detections = []
    chunking_param = request.detector_params.get("chunking") if request.detector_params else None
    if chunking_param and isinstance(chunking_param, dict):
        chunking_strategy = chunking_param.get("strategy")
        chunking_config = {k: v for k, v in chunking_param.items() if k != "strategy"}
    else:
        chunking_strategy = None
        chunking_config = {}
    print(f"DEBUG: Chunking strategy: {chunking_strategy}, config: {chunking_config}")
    for idx, content in enumerate(request.contents):
        message_detections = []
        print(f"DEBUG: Processing content {idx}: '{content}'")
        if chunking_strategy:
            chunker = chunker_registry.get(chunking_strategy)
            if not chunker:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unknown chunking strategy: {chunking_strategy}. Available: {chunker_registry.list_names()}"
                )
            chunks = chunker.chunk(content, **chunking_config)
            print(f"DEBUG: Chunks produced: {len(chunks)}")
            for chunk_idx, (chunk_text, start_pos, _) in enumerate(chunks):
                print(f"DEBUG: Chunk {chunk_idx}: '{chunk_text}' (start: {start_pos})")
                for detector_kind, detector_registry in app.get_all_detectors().items():
                    if not isinstance(detector_registry, BaseDetectorRegistry):
                        raise TypeError(f"Detector {detector_kind} is not a valid BaseDetectorRegistry")
                    if detector_kind in request.detector_params:
                        try:
                            chunk_detections = detector_registry.handle_request(chunk_text, request.detector_params)
                            for detection in chunk_detections:
                                detection.start += start_pos
                                detection.end += start_pos
                            message_detections += chunk_detections
                        except HTTPException as e:
                            raise e
                        except Exception as e:
                            raise HTTPException(status_code=500) from e
        else:
            print("DEBUG: No chunking, processing full content")
            for detector_kind, detector_registry in app.get_all_detectors().items():
                if not isinstance(detector_registry, BaseDetectorRegistry):
                    raise TypeError(f"Detector {detector_kind} is not a valid BaseDetectorRegistry")
                if detector_kind in request.detector_params:
                    try:
                        message_detections += detector_registry.handle_request(content, request.detector_params)
                    except HTTPException as e:
                        raise e
                    except Exception as e:
                        raise HTTPException(status_code=500) from e
        print(f"DEBUG: Total detections for content {idx}: {len(message_detections)}")
        detections.append(message_detections)
    return ContentsAnalysisResponse(root=detections)


@app.get("/registry")
def get_registry():
    result = {}
    for detector_type, detector_registry in app.get_all_detectors().items():
        if not isinstance(detector_registry, BaseDetectorRegistry):
            raise TypeError(f"Detector {detector_type} is not a valid BaseDetectorRegistry")
        result[detector_type] = {}
        for detector_name, detector_fn in detector_registry.get_registry().items():
            result[detector_type][detector_name] = detector_fn.__doc__
    return result