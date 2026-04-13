from typing import Literal

from pydantic import BaseModel


JobState = Literal[
    "uploaded",
    "parsing",
    "preprocessing",
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


class JobStatusResponse(BaseModel):
    job_id: str
    state: JobState
    created_at: str
    updated_at: str
    error: str | None = None
    history: list[HistoryEntry]
    artifacts: list[ArtifactStatus]


class ArtifactsResponse(BaseModel):
    job_id: str
    state: JobState
    artifacts: list[ArtifactStatus]
