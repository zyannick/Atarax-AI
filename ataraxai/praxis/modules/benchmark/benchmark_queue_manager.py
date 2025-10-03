import asyncio
import json
import logging
import threading
import traceback
import uuid
from dataclasses import asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from ataraxai.hegemonikon_py import HegemonikonBenchmarkResult  # type: ignore
from ataraxai.hegemonikon_py import ( # type: ignore
    HegemonikonLlamaBenchmarker,  # type: ignore; type: ignore
)
from ataraxai.praxis.utils.ataraxai_logger import AtaraxAILogger
from ataraxai.praxis.utils.background_task_manager import BackgroundTaskManager
from ataraxai.praxis.utils.configs.config_schemas.benchmarker_config_schema import (
    BenchmarkParams,
    BenchmarkResult,
    QuantizedModelInfo,
)
from ataraxai.praxis.utils.configs.config_schemas.llama_config_schema import (
    LlamaModelParams,
)


class BenchmarkJobStatus(str, Enum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class BenchmarkJob(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    model_info: QuantizedModelInfo
    benchmark_params: BenchmarkParams
    llama_model_params: LlamaModelParams
    status: BenchmarkJobStatus = BenchmarkJobStatus.QUEUED
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    benchmark_result: Optional[BenchmarkResult] = None
    error_message: Optional[str] = None

    class Config:
        use_enum_values = True


class BenchmarkQueueManager:
    def __init__(
        self,
        logger: logging.Logger,
        max_concurrent: int = 1,
        persistence_file: Optional[Path] = None,
    ):
        """
        Initialize a BenchmarkQueueManager instance.

        Args:
            logger (logging.Logger): Logger instance to use.
            max_concurrent (int): Maximum number of concurrent benchmark jobs. Defaults to 1.
            persistence_file (Optional[Path]): Path to the file used for persisting job state. If None, persistence is disabled.

        Attributes:
            logger (logging.Logger): Logger for the manager.
            max_concurrent (int): Maximum concurrent jobs allowed.
            persistence_file (Optional[Path]): File path for job persistence.
            _queue (List[BenchmarkJob]): Queue of pending benchmark jobs.
            _running (Dict[str, BenchmarkJob]): Dictionary of currently running jobs.
            _completed (Dict[str, BenchmarkJob]): Dictionary of completed jobs.
            _lock (threading.RLock): Reentrant lock for thread safety.
            _worker_task (Optional[asyncio.Task[Any]]): Async worker task for job processing.
            _shutdown_event (asyncio.Event): Event to signal shutdown.
            _job_added_event (asyncio.Event): Event to signal a new job has been added.
        """
        self.logger = logger
        self.max_concurrent = max(1, max_concurrent)
        self.persistence_file = persistence_file
        self._queue: List[BenchmarkJob] = []
        self._running: Dict[str, BenchmarkJob] = {}
        self._completed: Dict[str, BenchmarkJob] = {}
        self._lock = threading.RLock()

        self._worker_task: Optional[asyncio.Task[Any]] = None
        self._shutdown_event = asyncio.Event()
        self._job_added_event = asyncio.Event()

        self._load_persisted_jobs()

        self._runners: Dict[str, HegemonikonLlamaBenchmarker] = {}

    def enqueue_job(
        self,
        model_info: QuantizedModelInfo,
        benchmark_params: BenchmarkParams,
        llama_model_params: LlamaModelParams,
    ) -> str:
        """
        Enqueues a new benchmark job into the queue.

        Args:
            model_info (QuantizedModelInfo): Information about the quantized model to be benchmarked.
            benchmark_params (BenchmarkParams): Parameters for the benchmark job.
            llama_model_params (LlamaModelParams): Parameters specific to the Llama model.

        Returns:
            str: The unique identifier of the enqueued job.
        """
        job = BenchmarkJob(
            model_info=model_info,
            benchmark_params=benchmark_params,
            llama_model_params=llama_model_params,
        )
        with self._lock:
            self._queue.append(job)
            self._persist_jobs()
        self._job_added_event.set()
        self.logger.info(f"Enqueued job {job.id} for model {job.model_info.model_id}")
        return job.id

    def get_job(self, job_id: str) -> Optional[BenchmarkJob]:
        """
        Retrieve a benchmark job by its ID from the running, completed, or queued jobs.

        Args:
            job_id (str): The unique identifier of the benchmark job.

        Returns:
            Optional[BenchmarkJob]: The BenchmarkJob instance if found in running, completed, or queued jobs;
            otherwise, returns None.
        """
        with self._lock:
            if job_id in self._running:
                return self._running[job_id]
            if job_id in self._completed:
                return self._completed[job_id]
            for job in self._queue:
                if job.id == job_id:
                    return job
        return None

    async def cancel_job(self, job_id: str) -> bool:
        """
        Cancels a job in the queue with the specified job ID.

        Args:
            job_id (str): The unique identifier of the job to cancel.

        Returns:
            bool: True if the job was found and cancelled, False otherwise.

        Side Effects:
            - Updates the job status to CANCELLED.
            - Sets the job's completion time to the current time.
            - Moves the cancelled job from the queue to the completed jobs dictionary.
            - Persists the updated job states.
            - Logs the cancellation event.
        """
        self.logger.info(f"Attempting to cancel job {job_id}")
        with self._lock:
            for i, job in enumerate(self._queue):
                if job.id == job_id:
                    job.status = BenchmarkJobStatus.CANCELLED
                    job.completed_at = datetime.now().isoformat()
                    cancelled_job = self._queue.pop(i)
                    self._completed[job_id] = cancelled_job
                    self.logger.info(f"Cancelled queued job {job_id}")
                    self._persist_jobs()
                    return True
            
            if job_id in self._running:
                if job_id in self._runners:
                    self._runners[job_id].request_cancellation()
                    self.logger.info(
                        f"Requested cancellation for running job {job_id}. "
                        "Will stop at next checkpoint."
                    )
                    return True
                else:
                    self.logger.warning(
                        f"Job {job_id} is running but runner not found"
                    )
                    return False
        return False

    def get_queue_status(self) -> Dict[str, Any]:
        """
        Retrieves the current status of the job queue.

        Returns:
            Dict[str, Any]: A dictionary containing:
                - 'queued_count' (int): Number of jobs currently queued.
                - 'running_count' (int): Number of jobs currently running.
                - 'completed_count' (int): Number of jobs that have completed.
                - 'is_worker_running' (bool): Whether the worker is currently running.
                - 'running_jobs' (List[dict]): List of serialized representations of running jobs.
        """
        with self._lock:
            return {
                "queued_count": len(self._queue),
                "running_count": len(self._running),
                "completed_count": len(self._completed),
                "is_worker_running": self.is_worker_running(),
                "running_jobs": [job.model_dump() for job in self._running.values()],
            }

    def clear_completed_jobs(self) -> int:
        """
        Clears all completed jobs from the internal completed jobs list.

        Returns:
            int: The number of completed jobs that were cleared.
        """
        with self._lock:
            cleared_count = len(self._completed)
            self._completed.clear()
            self.logger.info(f"Cleared {cleared_count} completed jobs")
            self._persist_jobs()
        return cleared_count

    def is_worker_running(self) -> bool:
        """
        Check if the worker task is currently running.

        Returns:
            bool: True if the worker task exists and is not completed, False otherwise.
        """
        return self._worker_task is not None and not self._worker_task.done()

    async def start_worker(self):
        """
        Starts the benchmark queue worker asynchronously.

        Checks if the worker is already running; if so, logs a warning and returns.
        Otherwise, logs the start event, clears the shutdown event, and creates an asyncio task
        to run the worker loop.

        Returns:
            None
        """
        if self.is_worker_running():
            self.logger.warning("Worker is already running.")
            return
        self.logger.info("Starting benchmark queue worker.")
        self._shutdown_event.clear()
        self._worker_task = asyncio.create_task(self._worker_loop())

    async def stop_worker(self):
        """
        Gracefully stops the benchmark queue worker if it is running.

        This method sets the shutdown and job added events to signal the worker to stop.
        If the worker task is active, it waits for up to 10 seconds for the task to finish.
        If the worker does not stop within the timeout, it cancels the task.
        Handles cases where the worker is not running or is already stopping.

        Raises:
            asyncio.TimeoutError: If the worker does not stop within the timeout period.
            asyncio.CancelledError: If the worker task is cancelled during shutdown.
        """
        if not self.is_worker_running() or self._shutdown_event.is_set():
            self.logger.info("Worker is not running or already stopping.")
            return

        self.logger.info("Stopping benchmark queue worker...")
        self._shutdown_event.set()
        self._job_added_event.set()

        if self._worker_task:
            try:
                await asyncio.wait_for(self._worker_task, timeout=10.0)
            except asyncio.TimeoutError:
                self.logger.warning("Worker did not stop gracefully, cancelling.")
                self._worker_task.cancel()
            except asyncio.CancelledError:
                pass

    async def _worker_loop(self):
        """
        Asynchronous worker loop that continuously processes jobs from the queue.

        The loop initializes a benchmark runner and waits for jobs to be added to the queue.
        If no job is available, it waits for a job to be added or times out after 1 second.
        When a job is available, it moves the job to the running state and processes it.
        The loop exits gracefully when the shutdown event is set.

        Returns:
            None
        """
        while not self._shutdown_event.is_set():
            job = self._get_next_job()
            if not job:
                try:
                    self._job_added_event.clear()
                    await asyncio.wait_for(self._job_added_event.wait(), timeout=1.0)
                    continue
                except asyncio.TimeoutError:
                    continue

            self._move_job_to_running(job)
            await self._process_job(job)  # type: ignore

        self.logger.info("Worker loop has gracefully shut down.")

    def _get_next_job(self) -> Optional[BenchmarkJob]:
        """
        Retrieves the next job from the benchmark queue if available and if the number of running jobs is below the maximum concurrency limit.

        Returns:
            Optional[BenchmarkJob]: The next job to process if conditions are met; otherwise, None.
        """
        with self._lock:
            if self._queue and len(self._running) < self.max_concurrent:
                return self._queue.pop(0)
        return None

    def _move_job_to_running(self, job: BenchmarkJob):
        """
        Moves the specified BenchmarkJob to the running state.

        This method updates the job's status to RUNNING, sets its start time to the current time,
        adds it to the running jobs dictionary, logs the start event, and persists the job states.

        Args:
            job (BenchmarkJob): The job to move to the running state.
        """
        with self._lock:
            job.status = BenchmarkJobStatus.RUNNING
            job.started_at = datetime.now().isoformat()
            self._running[job.id] = job
            self.logger.info(
                f"Started job {job.id} for model {job.model_info.model_id}"
            )
            self._persist_jobs()

    async def _process_job(
        self, job: BenchmarkJob
    ):
        """
        Asynchronously processes a benchmark job using the provided runner.

        Executes the benchmark for a single model in a separate thread, updates the job's result and status,
        handles exceptions, and manages job completion and persistence.

        Args:
            job (BenchmarkJob): The benchmark job to process.
            runner (HegemonikonLlamaBenchmarker): The runner responsible for executing the benchmark.

        Side Effects:
            - Updates the job's result, status, error message, and completion time.
            - Logs job completion or failure.
            - Manages internal job tracking and persistence.
            - Signals job completion via an event.

        Exceptions:
            - Catches and logs any exceptions raised during benchmarking.
        """
        runner = HegemonikonLlamaBenchmarker()
        with self._lock:
            self._runners[job.id] = runner

        try:
            benchmark_metric_cpp: HegemonikonBenchmarkResult = await asyncio.to_thread(
                runner.benchmark_single_model,
                job.model_info.to_hegemonikon(),
                job.benchmark_params.to_hegemonikon(),
                job.llama_model_params.to_hegemonikon(),
            )

            with self._lock:
                benchmark_metric = BenchmarkResult.from_hegemonikon(benchmark_metric_cpp)
                job.benchmark_result = benchmark_metric
                
                if not benchmark_metric_cpp.metrics.success and \
                   "cancelled" in benchmark_metric_cpp.metrics.errorMessage.lower():
                    job.status = BenchmarkJobStatus.CANCELLED
                    self.logger.info(f"Job {job.id} was cancelled during execution.")
                else:
                    job.status = BenchmarkJobStatus.COMPLETED
                    self.logger.info(f"Completed job {job.id} successfully.")

        except Exception as e:
            self.logger.error(f"Job {job.id} failed: {e}", exc_info=True)
            with self._lock:
                job.status = BenchmarkJobStatus.FAILED
                job.error_message = str(e)

        finally:
            with self._lock:
                job.completed_at = datetime.now().isoformat()
                self._running.pop(job.id, None)
                self._completed[job.id] = job
                self._persist_jobs()
            self._job_added_event.set()

    def _persist_jobs(self):
        """
        Persists the current state of job queues (queued, running, completed) to a file.

        If a persistence file is configured, this method serializes the jobs in each queue
        using their `model_dump()` method and writes the resulting data as JSON to the file.
        Thread safety is ensured using a lock during serialization. If an error occurs during
        persistence, it is logged.

        Raises:
            Logs an error if persisting jobs fails.
        """
        if not self.persistence_file:
            return
        try:
            with self._lock:
                data = {
                    "queued": [job.model_dump() for job in self._queue],
                    "running": [job.model_dump() for job in self._running.values()],
                    "completed": [job.model_dump() for job in self._completed.values()],
                }
            with self.persistence_file.open("w") as f:
                json.dump(data, f, indent=4)
        except Exception as e:
            self.logger.debug(traceback.format_exc())
            self.logger.error(f"Failed to persist jobs: {e}")

    def _load_persisted_jobs(self):
        """
        Loads persisted benchmark jobs from the persistence file into the queue manager.

        This method reads job data from the specified persistence file, if it exists,
        and restores the state of queued, completed, and previously running jobs.
        - Queued jobs are loaded into the queue.
        - Completed jobs are loaded into the completed jobs dictionary.
        - Previously running jobs are re-queued with their status reset to QUEUED and
          their start time cleared.

        Logs the number of loaded queued and completed jobs, and any errors encountered
        during the loading process.

        Raises:
            Logs exceptions encountered during file reading or data parsing.
        """
        if not self.persistence_file or not self.persistence_file.exists():
            return
        try:
            data = json.loads(self.persistence_file.read_text())
            with self._lock:
                self._queue = [
                    BenchmarkJob.model_validate(j) for j in data.get("queued", [])
                ]
                self._completed = {
                    j["id"]: BenchmarkJob.model_validate(j)
                    for j in data.get("completed", [])
                }

                for job_data in data.get("running", []):
                    job = BenchmarkJob.model_validate(job_data)
                    job.status = BenchmarkJobStatus.QUEUED
                    job.started_at = None
                    self._queue.append(job)
                    self.logger.info(f"Re-queued previously running job {job.id}")
            self.logger.info(
                f"Loaded {len(self._queue)} queued and {len(self._completed)} completed jobs."
            )
        except Exception as e:
            self.logger.error(f"Failed to load persisted jobs: {e}")
