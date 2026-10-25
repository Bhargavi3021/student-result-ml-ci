
import logging
import sys
import time
import uuid

from fastapi import FastAPI, Request, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Configure application logging
logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger("student-result-api")

app = FastAPI(title="Student Result Prediction API")


# Request logging: request ID, HTTP status and response duration
@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    start_time = time.perf_counter()

    try:
        response = await call_next(request)
        duration_ms = (time.perf_counter() - start_time) * 1000

        level = (
            logging.WARNING
            if response.status_code >= 400
            else logging.INFO
        )

        logger.log(
            level,
            "event=http_request request_id=%s method=%s path=%s "
            "status=%s duration_ms=%.2f",
            request_id,
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
        )

        response.headers["X-Request-ID"] = request_id
        return response

    except Exception:
        duration_ms = (time.perf_counter() - start_time) * 1000
        logger.exception(
            "event=http_request_failed request_id=%s "
            "method=%s path=%s duration_ms=%.2f",
            request_id,
            request.method,
            request.url.path,
            duration_ms,
        )
        raise


# Log invalid request fields without logging the submitted payload
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
):
    logger.warning(
        "event=validation_error path=%s error_count=%s fields=%s",
        request.url.path,
        len(exc.errors()),
        [str(error["loc"]) for error in exc.errors()],
    )

    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors()},
    )


# Demonstration input schema
class StudentInput(BaseModel):
    student_id: str = Field(min_length=1)
    marks: float = Field(ge=0, le=100)


@app.on_event("startup")
async def startup_event():
    logger.info("event=service_started service=student-result-api")
    # No model is loaded in this demonstration API.


@app.get("/")
async def root():
    return {
        "service": "student-result-api",
        "status": "ok",
    }


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.post("/predict")
async def predict(student: StudentInput):
    try:
        logger.info("event=prediction_started")

        # Demonstration rule only; NOT a trained ML model.
        result = "PASS" if student.marks >= 40 else "FAIL"

        logger.info("event=prediction_completed result=%s", result)

        return {
            "student_id": student.student_id,
            "marks": student.marks,
            "prediction": result,
        }

    except Exception:
        logger.exception("event=prediction_failed")
        raise HTTPException(
            status_code=500,
            detail="Prediction failed",
        )
