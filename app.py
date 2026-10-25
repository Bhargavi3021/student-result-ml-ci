
import logging
import sys
import time
import uuid

from fastapi import FastAPI, Request, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger("student-result-api")

app = FastAPI(title="Student Result Prediction API")


@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = str(uuid.uuid4())
    start_time = time.perf_counter()

    try:
        response = await call_next(request)
        duration_ms = (time.perf_counter() - start_time) * 1000

        logger.info(
            "event=request_completed request_id=%s method=%s path=%s status=%s duration_ms=%.2f",
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
            "event=request_failed request_id=%s path=%s duration_ms=%.2f",
            request_id,
            request.url.path,
            duration_ms,
        )
        raise


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
):
    logger.warning(
        "event=validation_failed path=%s error_count=%s fields=%s",
        request.url.path,
        len(exc.errors()),
        [str(error["loc"]) for error in exc.errors()],
    )

    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors()},
    )


class StudentInput(BaseModel):
    student_id: str = Field(min_length=1)
    marks: float = Field(ge=0, le=100)


@app.on_event("startup")
async def startup_event():
    logger.info(
        "event=service_started service=student-result-api"
    )


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
        logger.info(
            "event=prediction_started student_id=%s",
            student.student_id,
        )

        # Demonstration rule only; this is not a trained ML model.
        result = "PASS" if student.marks >= 40 else "FAIL"

        logger.info(
            "event=prediction_completed student_id=%s result=%s",
            student.student_id,
            result,
        )

        return {
            "student_id": student.student_id,
            "marks": student.marks,
            "prediction": result,
        }

    except Exception:
        logger.exception(
            "event=prediction_failed student_id=%s",
            student.student_id,
        )
        raise HTTPException(
            status_code=500,
            detail="Prediction failed",
        )
