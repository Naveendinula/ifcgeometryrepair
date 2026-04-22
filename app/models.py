from typing import Any, Literal

from pydantic import BaseModel, Field


JobState = Literal[
    "uploaded",
    "parsing",
    "preprocessing",
    "preflight",
    "internal_boundary",
    "external_candidates",
    "external_shell",
    "opening_integration",
    "gbxml_preflight",
    "classifying",
    "complete",
    "failed",
]


class HistoryEntry(BaseModel):
    state: JobState
    timestamp: str
    message: str


class ArtifactStatus(BaseModel):
    name: str
    available: bool
    size_bytes: int | None = None
    url: str | None = None


class JobCreatedResponse(BaseModel):
    job_id: str
    state: JobState
    created_at: str
    status_url: str
    artifacts_url: str


class RemovedSpaceRef(BaseModel):
    global_id: str | None = None
    express_id: int
    name: str | None = None
    entity_type: str = "IfcSpace"
    storey: dict[str, Any] | None = None


class JobDerivationResponse(BaseModel):
    parent_job_id: str
    root_job_id: str
    operation: Literal["remove_spaces", "resolve_space_clashes"]
    removed_space_count: int
    removed_spaces: list[RemovedSpaceRef] = Field(default_factory=list)
    resolved_clash_group_count: int | None = None
    resolved_clash_group_ids: list[str] = Field(default_factory=list)


class DerivedJobCreatedResponse(JobCreatedResponse):
    parent_job_id: str
    root_job_id: str
    removed_space_count: int
    resolved_clash_group_count: int | None = None


class RemoveSpacesRequest(BaseModel):
    space_global_ids: list[str] = Field(default_factory=list)
    space_express_ids: list[int] = Field(default_factory=list)


class ClashGroupResolutionRequest(BaseModel):
    clash_group_id: str
    remove_space_global_ids: list[str] = Field(default_factory=list)
    remove_space_express_ids: list[int] = Field(default_factory=list)


class ResolveSpaceClashesRequest(BaseModel):
    group_resolutions: list[ClashGroupResolutionRequest] = Field(default_factory=list)


class JobStatusResponse(BaseModel):
    job_id: str
    state: JobState
    created_at: str
    updated_at: str
    error: str | None = None
    history: list[HistoryEntry]
    artifacts: list[ArtifactStatus]
    derivation: JobDerivationResponse | None = None


class ArtifactsResponse(BaseModel):
    job_id: str
    state: JobState
    artifacts: list[ArtifactStatus]
