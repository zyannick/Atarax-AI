import asyncio
import json
import logging
import threading
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from venv import logger

from pydantic import BaseModel, Field, field_validator

from ataraxai.hegemonikon_py import (  # type: ignore
    HegemonikonBenchmarkMetrics,
    HegemonikonBenchmarkParams,
    HegemonikonBenchmarkResult,
    HegemonikonLlamaBenchmarker,
    HegemonikonLlamaModelParams,
    HegemonikonQuantizedModelInfo,
)
from ataraxai.praxis.utils.ataraxai_logger import AtaraxAILogger
from ataraxai.praxis.utils.configs.config_schemas.benchmarker_config_schema import (
    BenchmarkMetrics,
    BenchmarkParams,
    BenchmarkResult,
    QuantizedModelInfo,
)
from ataraxai.praxis.utils.configs.config_schemas.llama_config_schema import (
    LlamaModelParams,
)


class BenchmarkJobStatus(Enum):
    QUEUED = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()
    CANCELLED = auto()


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

    def __init__(
        self,
        logger=AtaraxAILogger().get_logger(),
        max_concurrent: int = 1,
        persistence_file: Optional[Path] = None,
    ):
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
        self.logger = logger or logging.getLogger(__name__)
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
                    self.logger.info(f"Cancelled job {job_id}")
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
            self.logger.info(f"Cleared {cleared_count} completed jobs")
            self._persist_jobs()

    async def start_worker(self):
        if self._worker_task and not self._worker_task.done():
            self.logger.warning("Worker already running")
            return

        self.logger.info("Starting benchmark queue worker")
        self._shutdown_event.clear()
        self._worker_task = asyncio.create_task(self._worker_loop())

    async def stop_worker(self):
        self.logger.info("Stopping benchmark queue worker")
        self._shutdown_event.set()

        if self._worker_task:
            try:
                await asyncio.wait_for(self._worker_task, timeout=30.0)
            except asyncio.TimeoutError:
                self.logger.warning("Worker did not stop gracefully, cancelling")
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
                self.logger.error(f"Unexpected error in worker loop: {e}")
                await asyncio.sleep(1)

        self.logger.info("Worker loop stopped")

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
            self.logger.info(
                f"Started job {job.id} for model {job.model_info.model_id}"
            )
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
            result: HegemonikonBenchmarkResult = await asyncio.to_thread(
                runner.benchmarkSingleModel,
                job.model_info.to_hegemonikon(),
                job.benchmark_params.to_hegemonikon(),
                job.llama_model_params.to_hegemonikon(),
            )

            with self._lock:
                job.result = BenchmarkResult(
                    model_id=job.model_info.model_id,
                    metrics=BenchmarkMetrics.from_dict(asdict(result.metrics)),
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
