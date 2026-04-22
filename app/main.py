from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from typing import Literal

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .config import Settings
from .ifc_editing import InvalidSpaceRemovalRequestError, InvalidSpaceResolutionRequestError
from .job_service import ArtifactNotFoundError, InvalidJobOperationError, JobNotFoundError, JobService
from .models import (
    ArtifactsResponse,
    DerivedJobCreatedResponse,
    JobCreatedResponse,
    JobStatusResponse,
    RemoveSpacesRequest,
    ResolveSpaceClashesRequest,
)


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
            exact_repair_mode=settings.exact_repair_mode,
            exact_repair_worker_binary=settings.exact_repair_worker_binary,
            shell_worker_binary=settings.shell_worker_binary,
            internal_boundary_thickness_threshold_m=settings.internal_boundary_thickness_threshold_m,
            alpha_wrap_alpha_m=settings.alpha_wrap_alpha_m,
            alpha_wrap_offset_m=settings.alpha_wrap_offset_m,
            preflight_clash_tolerance_m=settings.preflight_clash_tolerance_m,
            min_surface_area_threshold_m2=settings.min_surface_area_threshold_m2,
            gbxml_tolerance_m=settings.gbxml_tolerance_m,
            gbxml_emit_on_validation_failure=settings.gbxml_emit_on_validation_failure,
            gbxml_min_surface_area_threshold_m2=settings.gbxml_min_surface_area_threshold_m2,
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
    async def create_job(
        request: Request,
        file: UploadFile = File(...),
        external_shell_mode: Literal["alpha_wrap", "heuristic"] = Form("alpha_wrap"),
        internal_boundary_thickness_threshold_m: float | None = Form(None, gt=0.0),
        alpha_wrap_alpha_m: float | None = Form(None, gt=0.0),
        alpha_wrap_offset_m: float | None = Form(None, gt=0.0),
    ) -> JobCreatedResponse:
        try:
            payload = request.app.state.job_service.create_job(
                file,
                external_shell_mode=external_shell_mode,
                internal_boundary_thickness_threshold_m=internal_boundary_thickness_threshold_m,
                alpha_wrap_alpha_m=alpha_wrap_alpha_m,
                alpha_wrap_offset_m=alpha_wrap_offset_m,
            )
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

    @app.post("/jobs/{job_id}/rerun/remove-spaces", status_code=202, response_model=DerivedJobCreatedResponse)
    async def rerun_remove_spaces(
        request: Request,
        job_id: str,
        payload: RemoveSpacesRequest,
    ) -> DerivedJobCreatedResponse:
        try:
            result = request.app.state.job_service.create_remove_spaces_rerun(
                job_id,
                space_global_ids=payload.space_global_ids,
                space_express_ids=payload.space_express_ids,
            )
        except JobNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except InvalidJobOperationError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except InvalidSpaceRemovalRequestError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return DerivedJobCreatedResponse(**result)

    @app.post("/jobs/{job_id}/rerun/resolve-space-clashes", status_code=202, response_model=DerivedJobCreatedResponse)
    async def rerun_resolve_space_clashes(
        request: Request,
        job_id: str,
        payload: ResolveSpaceClashesRequest,
    ) -> DerivedJobCreatedResponse:
        try:
            result = request.app.state.job_service.create_resolve_space_clashes_rerun(
                job_id,
                group_resolutions=[resolution.model_dump() for resolution in payload.group_resolutions],
            )
        except JobNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except InvalidJobOperationError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except (InvalidSpaceRemovalRequestError, InvalidSpaceResolutionRequestError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return DerivedJobCreatedResponse(**result)

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
