from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(slots=True)
class Settings:
    jobs_root: Path = PROJECT_ROOT / "jobs"
    stage_delay_seconds: float = 0.20
    geometry_worker_binary: Path = PROJECT_ROOT / "worker" / "build" / "geometry_worker.exe"
    internal_boundary_thickness_threshold_m: float = 0.30
