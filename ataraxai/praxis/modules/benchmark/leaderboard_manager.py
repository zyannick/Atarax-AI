import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from ataraxai.praxis.modules.benchmark.benchmark_queue_manager import (
    BenchmarkJob,
    BenchmarkResult,
)
from ataraxai.praxis.utils.ataraxai_logger import AtaraxAILogger


class LeaderboardEntry(BaseModel):
    model_id: str
    quantization: str
    best_result: BenchmarkResult
    last_updated: str = Field(default_factory=lambda: datetime.now().isoformat())

    @property
    def entry_key(self) -> str:
        return f"{self.model_id}_{self.quantization}"


class LeaderboardManager:
    def __init__(
        self,
        persistence_file: Path,
        logger: Optional[logging.Logger] = None,
    ):
        self.logger = logger or AtaraxAILogger("LeaderboardManager").get_logger()
        self.persistence_file = persistence_file
        self._leaderboard: Dict[str, LeaderboardEntry] = {}
        self._load_from_disk()

    def add_benchmark_job(self, job: BenchmarkJob):
        if job.status != "COMPLETED" or not job.benchmark_result:
            self.logger.debug(
                f"Skipping leaderboard update for non-completed job {job.id}"
            )
            return

        result = job.benchmark_result
        if not hasattr(job.model_info, "quantization"):
            self.logger.warning(
                f"Job {job.id} model_info is missing 'quantization' attribute."
            )
            return

        entry_key = f"{job.model_info.model_id}_{job.model_info.quantisation_type}"

        if (
            entry_key not in self._leaderboard
            or result.metrics.avg_decode_time_ms
            > self._leaderboard[entry_key].best_result.metrics.avg_decode_time_ms
        ):
            self.logger.info(
                f"New top score for {entry_key}! "
                f"Decode Time: {result.metrics.avg_decode_time_ms:.2f} ms. Updating leaderboard."
            )
            new_entry = LeaderboardEntry(
                model_id=job.model_info.model_id,
                quantization=job.model_info.quantisation_type,
                best_result=result,
                last_updated=datetime.now().isoformat(),
            )
            self._leaderboard[entry_key] = new_entry
            self._save_to_disk()
        else:
            self.logger.info(
                f"Score for {entry_key} ({result.metrics.avg_decode_time_ms:.2f} ms) did not beat "
                f"existing record ({self._leaderboard[entry_key].best_result.metrics.avg_decode_time_ms:.2f} ms)."
            )

    def remove_entries_for_model(self, model_id: str) -> int:
        keys_to_remove = [
            key
            for key, entry in self._leaderboard.items()
            if entry.model_id == model_id
        ]

        if not keys_to_remove:
            self.logger.info(
                f"No leaderboard entries found for model_id '{model_id}'. Nothing to do."
            )
            return 0

        for key in keys_to_remove:
            del self._leaderboard[key]

        self.logger.info(
            f"Removed {len(keys_to_remove)} leaderboard entries for deleted model_id '{model_id}'."
        )
        self._save_to_disk()
        return len(keys_to_remove)

    def get_leaderboard(
        self,
        sort_by: Literal[
            "avg_decode_time_ms", "avg_ttft_ms", "avg_end_to_end_latency_ms"
        ] = "avg_decode_time_ms",
        limit: int = 20,
    ) -> List[LeaderboardEntry]:
        if not self._leaderboard:
            return []

        reverse_sort = sort_by == "avg_decode_time_ms"

        sorted_entries = sorted(
            self._leaderboard.values(),
            key=lambda entry: getattr(entry.best_result.metrics, sort_by),
            reverse=reverse_sort,
        )
        return sorted_entries[:limit]

    def clear_leaderboard(self):
        self.logger.info("Clearing all leaderboard entries.")
        self._leaderboard.clear()
        self._save_to_disk()

    def _save_to_disk(self):
        try:
            data_to_save = {
                key: entry.model_dump(mode="json")
                for key, entry in self._leaderboard.items()
            }
            self.persistence_file.parent.mkdir(parents=True, exist_ok=True)
            self.persistence_file.write_text(json.dumps(data_to_save, indent=2))
            self.logger.debug(f"Leaderboard saved to {self.persistence_file}")
        except Exception as e:
            self.logger.error(f"Failed to save leaderboard: {e}", exc_info=True)

    def _load_from_disk(self):
        if not self.persistence_file.exists():
            self.logger.info("No leaderboard file found. Starting fresh.")
            return

        try:
            data = json.loads(self.persistence_file.read_text())
            self._leaderboard = {
                key: LeaderboardEntry.model_validate(value)
                for key, value in data.items()
            }
            self.logger.info(
                f"Loaded {len(self._leaderboard)} entries from {self.persistence_file}"
            )
        except Exception as e:
            self.logger.error(
                f"Failed to load or parse leaderboard file: {e}", exc_info=True
            )
            self._leaderboard = {}
