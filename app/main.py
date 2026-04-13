from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .config import Settings
from .job_service import ArtifactNotFoundError, JobNotFoundError, JobService
from .models import ArtifactsResponse, JobCreatedResponse, JobStatusResponse


STATIC_DIR = Path(__file__).resolve().parent / "static"


def create_app(
    settings: Settings | None = None,
) -> FastAPI:
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        service = JobService(
            jobs_root=settings.jobs_root,
            stage_delay_seconds=settings.stage_delay_seconds,
            geometry_worker_binary=settings.geometry_worker_binary,
            internal_boundary_thickness_threshold_m=settings.internal_boundary_thickness_threshold_m,
        )
        service.start()
        app.state.job_service = service
        app.state.settings = settings
        yield
        service.stop()

    app = FastAPI(
        title="IFC Geometry Repair",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.post("/jobs", status_code=202, response_model=JobCreatedResponse)
    async def create_job(request: Request, file: UploadFile = File(...)) -> JobCreatedResponse:
        try:
            payload = request.app.state.job_service.create_job(file)
        finally:
            await file.close()
        return JobCreatedResponse(**payload)

    @app.get("/jobs/{job_id}", response_model=JobStatusResponse)
    async def get_job_status(request: Request, job_id: str) -> JobStatusResponse:
        try:
            payload = request.app.state.job_service.get_status(job_id)
        except JobNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return JobStatusResponse(**payload)

    @app.get("/jobs/{job_id}/artifacts", response_model=ArtifactsResponse)
    async def get_job_artifacts(request: Request, job_id: str) -> ArtifactsResponse:
        try:
            payload = request.app.state.job_service.list_artifacts(job_id)
        except JobNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return ArtifactsResponse(**payload)

    @app.get("/jobs/{job_id}/artifacts/{artifact_path:path}", name="download_artifact")
    async def download_artifact(request: Request, job_id: str, artifact_path: str) -> FileResponse:
        try:
            file_path = request.app.state.job_service.get_artifact_path(job_id, artifact_path)
        except (JobNotFoundError, ArtifactNotFoundError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return FileResponse(file_path, filename=file_path.name)

    return app


app = create_app()
