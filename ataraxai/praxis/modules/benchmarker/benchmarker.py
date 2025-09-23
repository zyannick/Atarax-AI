import asyncio
import json
import logging
import threading
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

from ataraxai.hegemonikon_py import (  # type: ignore
    HegemonikonBenchmarkMetrics,
    HegemonikonBenchmarkParams,
    HegemonikonBenchmarkResult,
    HegemonikonLlamaBenchmarker,
    HegemonikonLlamaModelParams,
    HegemonikonQuantizedModelInfo,
)
from ataraxai.praxis.utils.configs.config_schemas.llama_config_schema import (
    GenerationParams,
    LlamaModelParams,
)


class QuantizedModelInfo(BaseModel):
    model_id: str = Field(..., description="Unique identifier for the model.")
    local_path: str = Field(..., description="Local filesystem path to the model.")
    last_modified: str = Field(..., description="Timestamp of the last modification.")
    quantisation_type: str = Field(
        ..., description="Type of quantization applied to the model."
    )
    size_bytes: int = Field(..., description="Size of the model file in bytes.")

    @field_validator("size_bytes")
    def validate_size_bytes(cls, value: int) -> int:
        if value < 0:
            raise ValueError("Size in bytes must be non-negative.")
        return value

    @field_validator("local_path")
    def validate_local_path(cls, value: str) -> str:
        if not value:
            raise ValueError("Local path must be a non-empty string.")
        if not Path(value).exists():
            raise ValueError("Local path must point to an existing file.")
        return value

    def to_hegemonikon(self) -> HegemonikonQuantizedModelInfo:
        return HegemonikonQuantizedModelInfo.from_dict(self.model_dump())


class BenchmarkMetrics(BaseModel):
    load_time_ms: float = Field(
        ..., description="Time taken to load the model in milliseconds."
    )
    generation_time_ms: float = Field(
        ..., description="Time taken for generation in milliseconds."
    )
    total_time_ms: float = Field(
        ..., description="Total time taken for the benchmark in milliseconds."
    )
    tokens_generated: int = Field(
        ..., description="Number of tokens generated during the benchmark."
    )
    token_per_second: float = Field(..., description="Tokens generated per second.")
    error_message: str = Field(
        "", description="Error message if any error occurred during benchmarking."
    )
    memory_usage_mb: float = Field(
        ..., description="Memory usage in megabytes during the benchmark."
    )
    success: bool = Field(..., description="Indicates if the benchmark was successful.")

    generation_time_history_ms: List[float] = Field(
        default_factory=list,
        description="List of individual generation times in milliseconds.",
    )
    token_per_second_times_history_ms: List[float] = Field(
        default_factory=list,
        description="List of tokens per second recorded at different intervals.",
    )
    ttft_history_ms: List[float] = Field(
        default_factory=list,
        description="List of time to first token measurements in milliseconds.",
    )
    end_to_end_latency_history_ms: List[float] = Field(
        default_factory=list,
        description="List of end-to-end latency measurements in milliseconds.",
    )
    decode_times_ms: List[float] = Field(
        default_factory=list, description="List of decode times in milliseconds."
    )

    avg_ttft_ms: float = Field(
        0.0, description="Average time to first token in milliseconds."
    )
    avg_decode_time_ms: float = Field(
        0.0, description="Average decode time in milliseconds."
    )
    avg_end_to_end_time_latency_ms: float = Field(
        0.0, description="Average end-to-end latency in milliseconds."
    )

    p50_latency_ms: float = Field(
        0.0, description="50th percentile latency in milliseconds."
    )
    p95_latency_ms: float = Field(
        0.0, description="95th percentile latency in milliseconds."
    )
    p99_latency_ms: float = Field(
        0.0, description="99th percentile latency in milliseconds."
    )

    @field_validator(
        "load_time_ms", "generation_time_ms", "total_time_ms", "memory_usage_mb"
    )
    def validate_non_negative(cls, value: float) -> float:
        if value < 0:
            raise ValueError("Value must be non-negative.")
        return value

    def to_hegemonikon(self) -> HegemonikonBenchmarkMetrics:
        return HegemonikonBenchmarkMetrics.from_dict(self.model_dump())
    
    @classmethod
    def from_dict(cls, data: Dict) -> "BenchmarkMetrics":
        return cls(**data)


class BenchmarkParams(BaseModel):
    n_gpu_layers: int = Field(..., description="Number of GPU layers to use.")
    repetitions: int = Field(
        ..., description="Number of repetitions for the benchmark."
    )
    warmup: bool = Field(..., description="Whether to perform warmup runs.")
    generation_params: GenerationParams = Field(
        ..., description="Parameters for text generation."
    )

    @field_validator("n_gpu_layers", "repetitions")
    def validate_non_negative_int(cls, value: int) -> int:
        if value < 0:
            raise ValueError("Value must be a non-negative integer.")
        return value

    def to_hegemonikon(self) -> HegemonikonBenchmarkParams:
        return HegemonikonBenchmarkParams.from_dict(self.model_dump())


class BenchmarkResult(BaseModel):
    model_id: str = Field(..., description="Unique identifier for the model.")
    metrics: BenchmarkMetrics = Field(
        ..., description="Benchmark metrics for the model."
    )
    # benchmark_params: BenchmarkParams = Field(
    #     ..., description="Parameters used for benchmarking."
    # )
    # llama_model_params: LlamaModelParams = Field(
    #     ..., description="Llama model parameters used during benchmarking."
    # )

    @field_validator("model_id")
    def validate_model_id(cls, value: str) -> str:
        if not value:
            raise ValueError("Model ID must be a non-empty string.")
        return value


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class BenchmarkJobStatus(Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class BenchmarkJob(BaseModel):
    id: str = Field(..., description="Unique identifier for the benchmark job.")
    model_info: QuantizedModelInfo = Field(
        ..., description="Information about the quantized model to benchmark."
    )
    benchmark_params: BenchmarkParams = Field(
        ..., description="Parameters for the benchmark."
    )
    llama_model_params: LlamaModelParams = Field(
        ..., description="Llama model parameters to use during benchmarking."
    )
    status: BenchmarkJobStatus = Field(
        default=BenchmarkJobStatus.QUEUED,
        description="Current status of the benchmark job.",
    )
    created_at: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
        description="Timestamp when the job was created.",
    )
    started_at: Optional[str] = Field(
        default=None, description="Timestamp when the job started."
    )
    completed_at: Optional[str] = Field(
        default=None, description="Timestamp when the job completed."
    )
    result: Optional[BenchmarkResult] = Field(
        default=None, description="Result of the benchmark job."
    )
    error_message: Optional[str] = Field(
        default=None, description="Error message if the job failed."
    )

    @classmethod
    def from_dict(cls, data: Dict) -> "BenchmarkJob":
        job = cls(
            id=data["id"],
            model_info=QuantizedModelInfo(**data["model_info"]),
            benchmark_params=BenchmarkParams(**data["benchmark_params"]),
            llama_model_params=LlamaModelParams(**data["llama_model_params"]),
            status=BenchmarkJobStatus(data["status"]),
            created_at=data["created_at"],
            started_at=data.get("started_at"),
            completed_at=data.get("completed_at"),
            error_message=data.get("error_message"),
        )
        if data.get("result"):
            job.result = BenchmarkResult(**data["result"])
            pass
        return job


class BenchmarkQueueManager:

    def __init__(self, max_concurrent: int = 1, persistence_file: Optional[Path] = None):
        """
        Initializes the Benchmarker instance.

        Args:
            max_concurrent (int, optional): Maximum number of concurrent benchmark jobs. Defaults to 1.
            persistence_file (Optional[str], optional): Path to the file used for persisting job state. Defaults to None.

        Attributes:
            max_concurrent (int): Maximum number of concurrent jobs.
            persistence_file (Optional[Path]): Path object for the persistence file, if provided.
            _queue (List[BenchmarkJob]): Queue of pending benchmark jobs.
            _running (Dict[str, BenchmarkJob]): Dictionary of currently running jobs.
            _completed (Dict[str, BenchmarkJob]): Dictionary of completed jobs.
            _lock (threading.RLock): Reentrant lock for thread safety.
            _worker_task (Optional[asyncio.Task]): Async worker task for job processing.
            _shutdown_event (asyncio.Event): Event to signal shutdown.
            _job_added_event (asyncio.Event): Event to signal a new job has been added.
            _job_started_callbacks (List[Callable[[BenchmarkJob], None]]): Callbacks for job start events.
            _job_completed_callbacks (List[Callable[[BenchmarkJob], None]]): Callbacks for job completion events.
            _job_failed_callbacks (List[Callable[[BenchmarkJob], None]]): Callbacks for job failure events.

        Calls:
            _load_persisted_jobs(): Loads jobs from the persistence file if provided.
        """
        self.max_concurrent = max_concurrent
        self.persistence_file = persistence_file
        self._queue: List[BenchmarkJob] = []
        self._running: Dict[str, BenchmarkJob] = {}
        self._completed: Dict[str, BenchmarkJob] = {}
        self._lock = threading.RLock()

        self._worker_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()
        self._job_added_event = asyncio.Event()

        self._job_started_callbacks: List[Callable[[BenchmarkJob], None]] = []
        self._job_completed_callbacks: List[Callable[[BenchmarkJob], None]] = []
        self._job_failed_callbacks: List[Callable[[BenchmarkJob], None]] = []

        self._load_persisted_jobs()

    def add_job_started_callback(self, callback: Callable[[BenchmarkJob], None]):
        self._job_started_callbacks.append(callback)

    def add_job_completed_callback(self, callback: Callable[[BenchmarkJob], None]):
        self._job_completed_callbacks.append(callback)

    def add_job_failed_callback(self, callback: Callable[[BenchmarkJob], None]):
        self._job_failed_callbacks.append(callback)

    def enqueue_job(
        self,
        model_info: QuantizedModelInfo,
        benchmark_params: BenchmarkParams,
        llama_model_params: Any,
    ) -> str:
        job_id = str(uuid.uuid4())
        job = BenchmarkJob(
            id=job_id,
            model_info=model_info,
            benchmark_params=benchmark_params,
            llama_model_params=llama_model_params,
        )

        with self._lock:
            self._queue.append(job)
            self._persist_jobs()

        self._job_added_event.set()
        return job_id

    def get_job_status(self, job_id: str) -> Optional[BenchmarkJobStatus]:
        with self._lock:
            if job_id in self._running:
                return self._running[job_id].status

            if job_id in self._completed:
                return self._completed[job_id].status

            for job in self._queue:
                if job.id == job_id:
                    return job.status

        return None

    def get_job(self, job_id: str) -> Optional[BenchmarkJob]:
        with self._lock:
            if job_id in self._running:
                return self._running[job_id]
            if job_id in self._completed:
                return self._completed[job_id]
            for job in self._queue:
                if job.id == job_id:
                    return job
        return None

    def cancel_job(self, job_id: str) -> bool:
        with self._lock:
            for i, job in enumerate(self._queue):
                if job.id == job_id:
                    job.status = BenchmarkJobStatus.CANCELLED
                    cancelled_job = self._queue.pop(i)
                    self._completed[job_id] = cancelled_job
                    logger.info(f"Cancelled job {job_id}")
                    self._persist_jobs()
                    return True

        return False

    def get_queue_status(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "queued": len(self._queue),
                "running": len(self._running),
                "completed": len(self._completed),
                "queue_jobs": [
                    {"id": job.id, "model": job.model_info.model_id}
                    for job in self._queue
                ],
                "running_jobs": [
                    {
                        "id": job.id,
                        "model": job.model_info.model_id,
                        "started_at": job.started_at,
                    }
                    for job in self._running.values()
                ],
                "max_concurrent": self.max_concurrent,
            }

    def clear_completed_jobs(self):
        with self._lock:
            cleared_count = len(self._completed)
            self._completed.clear()
            logger.info(f"Cleared {cleared_count} completed jobs")
            self._persist_jobs()

    async def start_worker(self):
        if self._worker_task and not self._worker_task.done():
            logger.warning("Worker already running")
            return

        logger.info("Starting benchmark queue worker")
        self._shutdown_event.clear()
        self._worker_task = asyncio.create_task(self._worker_loop())

    async def stop_worker(self):
        logger.info("Stopping benchmark queue worker")
        self._shutdown_event.set()

        if self._worker_task:
            try:
                await asyncio.wait_for(self._worker_task, timeout=30.0)
            except asyncio.TimeoutError:
                logger.warning("Worker did not stop gracefully, cancelling")
                self._worker_task.cancel()
                try:
                    await self._worker_task
                except asyncio.CancelledError:
                    pass

    async def _worker_loop(self):
        runner = HegemonikonLlamaBenchmarker()

        while not self._shutdown_event.is_set():
            try:
                if not self._queue or len(self._running) >= self.max_concurrent:
                    self._job_added_event.clear()
                    await asyncio.wait_for(self._job_added_event.wait(), timeout=1.0)
                    continue

                job = self._get_next_job()
                if not job:
                    continue

                self._move_job_to_running(job)

                await self._process_job(job, runner)

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Unexpected error in worker loop: {e}")
                await asyncio.sleep(1)

        logger.info("Worker loop stopped")

    def _get_next_job(self) -> Optional[BenchmarkJob]:
        with self._lock:
            if self._queue and len(self._running) < self.max_concurrent:
                return self._queue.pop(0)
        return None

    def _move_job_to_running(self, job: BenchmarkJob):
        with self._lock:
            job.status = BenchmarkJobStatus.RUNNING
            job.started_at = datetime.now().isoformat()
            self._running[job.id] = job
            logger.info(f"Started job {job.id} for model {job.model_info.model_id}")
            self._persist_jobs()

        for callback in self._job_started_callbacks:
            try:
                callback(job)
            except Exception as e:
                logger.error(f"Error in job started callback: {e}")

    async def _process_job(
        self, job: BenchmarkJob, runner: HegemonikonLlamaBenchmarker
    ):
        try:
            result : HegemonikonBenchmarkResult = await asyncio.to_thread(
                runner.benchmarkSingleModel,
                job.model_info.to_hegemonikon(),
                job.benchmark_params.to_hegemonikon(),
                job.llama_model_params.to_hegemonikon(),
            )

            with self._lock:
                job.result = BenchmarkResult(
                    model_id=job.model_info.model_id,
                    metrics=BenchmarkMetrics.from_dict(
                        asdict(result.metrics)
                    ),
                )
                job.status = BenchmarkJobStatus.COMPLETED
                job.completed_at = datetime.now().isoformat()

                self._running.pop(job.id, None)
                self._completed[job.id] = job

                logger.info(f"Completed job {job.id} successfully")
                self._persist_jobs()

            for callback in self._job_completed_callbacks:
                try:
                    callback(job)
                except Exception as e:
                    logger.error(f"Error in job completed callback: {e}")

        except Exception as e:
            logger.error(f"Job {job.id} failed: {str(e)}")

            with self._lock:
                job.status = BenchmarkJobStatus.FAILED
                job.error_message = str(e)
                job.completed_at = datetime.now().isoformat()

                self._running.pop(job.id, None)
                self._completed[job.id] = job
                self._persist_jobs()

            for callback in self._job_failed_callbacks:
                try:
                    callback(job)
                except Exception as e:
                    logger.error(f"Error in job failed callback: {e}")

        self._job_added_event.set()

    def _persist_jobs(self):
        if not self.persistence_file:
            return

        try:
            with self._lock:
                data = {
                    "queued": [job.model_dump() for job in self._queue],
                    "running": [job.model_dump() for job in self._running.values()],
                    "completed": [job.model_dump() for job in self._completed.values()],
                    "last_updated": datetime.now().isoformat(),
                }

            with open(self.persistence_file, "w") as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            logger.error(f"Failed to persist jobs: {e}")

    def _load_persisted_jobs(self):
        if not self.persistence_file or not self.persistence_file.exists():
            return

        try:
            with open(self.persistence_file, "r") as f:
                data = json.load(f)

            with self._lock:
                for job_data in data.get("queued", []):
                    try:
                        job = BenchmarkJob.from_dict(job_data)
                        self._queue.append(job)
                    except Exception as e:
                        logger.error(f"Failed to load queued job: {e}")

                for job_data in data.get("completed", []):
                    try:
                        job = BenchmarkJob.from_dict(job_data)
                        self._completed[job.id] = job
                    except Exception as e:
                        logger.error(f"Failed to load completed job: {e}")

                for job_data in data.get("running", []):
                    try:
                        job = BenchmarkJob.from_dict(job_data)
                        job.status = BenchmarkJobStatus.QUEUED
                        job.started_at = None
                        self._queue.append(job)
                        logger.info(f"Re-queued previously running job {job.id}")
                    except Exception as e:
                        logger.error(f"Failed to re-queue running job: {e}")

            logger.info(
                f"Loaded {len(self._queue)} queued and {len(self._completed)} completed jobs from persistence"
            )

        except Exception as e:
            logger.error(f"Failed to load persisted jobs: {e}")
