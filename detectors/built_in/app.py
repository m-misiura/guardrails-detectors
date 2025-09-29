import re
from contextlib import asynccontextmanager
from typing import List, Tuple

from base_detector_registry import BaseDetectorRegistry
from fastapi import HTTPException
from file_type_detectors import FileTypeDetectorRegistry
from prometheus_fastapi_instrumentator import Instrumentator
from regex_detectors import RegexDetectorRegistry

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


# registry : dict[str, BaseDetectorRegistry] = {
#     "regex": RegexDetectorRegistry(),
#     "file_type": FileTypeDetectorRegistry(),
# }

SENTENCE_PATTERN = re.compile(r'[.!?]+(?=\s+[A-Z]|$)')

def split_sentences(text: str) -> List[Tuple[str, int, int]]:
    """Simple sentence splitting. Returns (sentence, start, end) tuples."""
    if not text.strip():
        return []
    sentences = []
    last_end = 0
    for match in SENTENCE_PATTERN.finditer(text):
        raw_sentence = text[last_end:match.end()]
        stripped = raw_sentence.strip()
        if stripped:
            leading_spaces = len(raw_sentence) - len(raw_sentence.lstrip())
            start_pos = last_end + leading_spaces
            end_pos = start_pos + len(stripped)
            sentences.append((stripped, start_pos, end_pos))
        last_end = match.end()
    if last_end < len(text):
        raw_sentence = text[last_end:]
        stripped = raw_sentence.strip()
        if stripped:
            leading_spaces = len(raw_sentence) - len(raw_sentence.lstrip())
            start_pos = last_end + leading_spaces
            end_pos = start_pos + len(stripped)
            sentences.append((stripped, start_pos, end_pos))
    return sentences or [(text.strip(), 0, len(text))] if text.strip() else []


@app.post("/api/v1/text/contents", response_model=ContentsAnalysisResponse)
def detect_content(request: ContentAnalysisHttpRequest):
    detections = []
    chunking = request.detector_params.get("chunking") if request.detector_params else None
    for content_idx, content in enumerate(request.contents):
        message_detections = []
        if chunking == "sentence":
            sentences = split_sentences(content)
            for sentence_idx, (sentence, start_pos, _) in enumerate(sentences):
                for detector_kind, detector_registry in app.get_all_detectors().items():
                    if not isinstance(detector_registry, BaseDetectorRegistry):
                        raise TypeError(f"Detector {detector_kind} is not a valid BaseDetectorRegistry")
                    if detector_kind in request.detector_params:
                        try:
                            sentence_detections = detector_registry.handle_request(sentence, request.detector_params)
                            for detection in sentence_detections:
                                original_start = detection.start
                                original_end = detection.end
                                detection.start += start_pos
                                detection.end += start_pos
                            message_detections += sentence_detections
                        except HTTPException as e:
                            raise e
                        except Exception as e:
                            raise HTTPException(status_code=500) from e
        else:
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