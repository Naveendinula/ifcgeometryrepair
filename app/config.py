from dataclasses import dataclass
from pathlib import Path
from typing import Literal


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EXACT_REPAIR_BINARY = PROJECT_ROOT / "worker" / "build" / "geometry_worker.exe"
DEFAULT_SHELL_WORKER_BINARY = PROJECT_ROOT / "worker" / "build" / "shell_worker.exe"


@dataclass(slots=True)
class Settings:
    jobs_root: Path = PROJECT_ROOT / "jobs"
    stage_delay_seconds: float = 0.20
    geometry_worker_binary: Path | None = None
    exact_repair_mode: Literal["disabled", "preferred"] = "preferred"
    exact_repair_worker_binary: Path | None = None
    shell_worker_binary: Path | None = None
    internal_boundary_thickness_threshold_m: float = 0.30
    preflight_clash_tolerance_m: float = 0.01

    def __post_init__(self) -> None:
        resolved_exact_repair_binary = self.exact_repair_worker_binary or self.geometry_worker_binary or DEFAULT_EXACT_REPAIR_BINARY
        self.exact_repair_worker_binary = resolved_exact_repair_binary
        # Keep the legacy alias populated for one compatibility cycle.
        self.geometry_worker_binary = resolved_exact_repair_binary
        self.shell_worker_binary = self.shell_worker_binary or DEFAULT_SHELL_WORKER_BINARY
