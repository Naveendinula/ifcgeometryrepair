import { DebugViewer } from "./viewer.js";


const form = document.querySelector("#upload-form");
const fileInput = document.querySelector("#file-input");
const selectedFileNameNode = document.querySelector("#selected-file-name");
const emptyUploadButton = document.querySelector("#empty-upload-button");
const submitButton = document.querySelector("#submit-button");
const flashMessage = document.querySelector("#flash-message");

const jobIdNode = document.querySelector("#job-id");
const jobIdRailNode = document.querySelector("#job-id-rail");
const jobStateNode = document.querySelector("#job-state");
const jobStateRailNode = document.querySelector("#job-state-rail");
const jobUpdatedNode = document.querySelector("#job-updated");
const jobUpdatedRailNode = document.querySelector("#job-updated-rail");
const jobStatePillNode = document.querySelector("#job-state-pill");
const jobStateRailPillNode = document.querySelector("#job-state-rail-pill");
const jobRunCardNode = document.querySelector("#job-run-card");
const jobErrorNode = document.querySelector("#job-error");
const jobHistoryNode = document.querySelector("#job-history");

const statusLinkNode = document.querySelector("#status-link");
const outputLinkNode = document.querySelector("#output-link");
const viewerManifestLinkNode = document.querySelector("#viewer-manifest-link");

const reportSchemaNode = document.querySelector("#report-schema");
const spacesCountNode = document.querySelector("#spaces-count");
const openingsCountNode = document.querySelector("#openings-count");
const missingCountNode = document.querySelector("#missing-count");
const invalidCountNode = document.querySelector("#invalid-count");
const workerBackendNode = document.querySelector("#worker-backend");
const validNormalizedCountNode = document.querySelector("#valid-normalized-count");
const invalidNormalizedCountNode = document.querySelector("#invalid-normalized-count");
const geometryUnitNode = document.querySelector("#geometry-unit");

const layerRawCountNode = document.querySelector("#layer-raw-count");
const layerNormalizedCountNode = document.querySelector("#layer-normalized-count");
const layerOpeningsCountNode = document.querySelector("#layer-openings-count");
const layerFailedCountNode = document.querySelector("#layer-failed-count");

const artifactCountNode = document.querySelector("#artifact-count");
const artifactListNode = document.querySelector("#artifact-list");
const spacesObjectCountNode = document.querySelector("#spaces-object-count");
const openingsObjectCountNode = document.querySelector("#openings-object-count");
const failedObjectCountNode = document.querySelector("#failed-object-count");

const viewer = new DebugViewer({
  panel: document.querySelector("#viewer-panel"),
  manifestLink: viewerManifestLinkNode,
  canvas: document.querySelector("#viewer-canvas"),
  emptyState: document.querySelector("#viewer-empty"),
  emptyStateMessage: document.querySelector("#viewer-empty-message"),
  selectionEmpty: document.querySelector("#viewer-selection-empty"),
  metadataList: document.querySelector("#viewer-metadata"),
  artifactLinks: document.querySelector("#viewer-artifact-links"),
  spaceList: document.querySelector("#viewer-space-list"),
  openingList: document.querySelector("#viewer-opening-list"),
  failedList: document.querySelector("#viewer-failed-list"),
  toggleRaw: document.querySelectorAll('[data-layer-toggle="raw"]'),
  toggleNormalized: document.querySelectorAll('[data-layer-toggle="normalized"]'),
  toggleOpenings: document.querySelectorAll('[data-layer-toggle="openings"]'),
  toggleFailed: document.querySelectorAll('[data-layer-toggle="failed"]'),
  resetViewButtons: [document.querySelector("#reset-view-button")],
  fitViewButtons: [document.querySelector("#fit-view-button")],
  onError: (message) => setFlash(message, true),
});

let pollHandle = null;
let renderedReportJobId = null;
let renderedReport = null;

const terminalStates = new Set(["complete", "failed"]);

fileInput.addEventListener("change", () => updateSelectedFileName());
emptyUploadButton.addEventListener("click", () => fileInput.click());

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const file = fileInput.files[0];
  if (!file) {
    setFlash("Choose an IFC file first.", true);
    return;
  }

  stopPolling();
  submitButton.disabled = true;
  renderedReportJobId = null;
  renderedReport = null;
  clearWorkspaceForNewRun();
  setFlash("Uploading file and creating job...", false);

  const payload = new FormData();
  payload.append("file", file);

  try {
    const response = await fetch("/jobs", {
      method: "POST",
      body: payload,
    });
    const body = await response.json();
    if (!response.ok) {
      throw new Error(body.detail || "Job creation failed.");
    }

    statusLinkNode.href = body.status_url;
    statusLinkNode.classList.remove("hidden");
    updateJobHeader(body.job_id, body.state, body.created_at);
    setFlash(`Job ${body.job_id} created.`, false);

    await refreshStatus(body.job_id);
    startPolling(body.job_id);
  } catch (error) {
    setFlash(error.message, true);
  } finally {
    submitButton.disabled = false;
  }
});

initializeWorkspace();

function initializeWorkspace() {
  updateSelectedFileName();
  updateJobHeader(null, "idle", null);
  renderHistory([]);
  renderError("");
  renderArtifacts([]);
  resetSummary();
  hideQuickLinks();
  viewer.clear("Use the import control above to create a job and populate the 3D workspace.");
  setFlash("", false);
}

function clearWorkspaceForNewRun() {
  renderHistory([]);
  renderError("");
  renderArtifacts([]);
  resetSummary();
  hideQuickLinks();
  viewer.clear("Viewer layers will appear once preprocessing completes.");
}

function updateSelectedFileName() {
  selectedFileNameNode.textContent = fileInput.files[0]?.name || "No file selected";
}

function startPolling(jobId) {
  stopPolling();
  pollHandle = window.setInterval(() => {
    refreshStatus(jobId).catch((error) => {
      setFlash(error.message, true);
      stopPolling();
    });
  }, 1000);
}

function stopPolling() {
  if (pollHandle !== null) {
    window.clearInterval(pollHandle);
    pollHandle = null;
  }
}

async function refreshStatus(jobId) {
  const response = await fetch(`/jobs/${jobId}`);
  const body = await response.json();
  if (!response.ok) {
    throw new Error(body.detail || "Could not load job status.");
  }

  updateJobHeader(body.job_id, body.state, body.updated_at);
  renderHistory(body.history);
  renderError(body.error);
  await refreshArtifacts(jobId);

  if (body.state === "complete") {
    const report = await refreshReport(jobId);
    await viewer.loadJob(jobId, report);
  } else if (body.state === "failed") {
    viewer.clear("Viewer artifacts are not available for this failed run.");
  }

  if (terminalStates.has(body.state)) {
    stopPolling();
  }
}

async function refreshArtifacts(jobId) {
  const response = await fetch(`/jobs/${jobId}/artifacts`);
  const body = await response.json();
  if (!response.ok) {
    throw new Error(body.detail || "Could not load artifacts.");
  }

  renderArtifacts(body.artifacts);

  const artifactNames = new Set(body.artifacts.map((artifact) => artifact.name));
  statusLinkNode.href = `/jobs/${jobId}`;
  statusLinkNode.classList.remove("hidden");

  if (artifactNames.has("output.json")) {
    outputLinkNode.href = `/jobs/${jobId}/artifacts/output.json`;
    outputLinkNode.classList.remove("hidden");
  } else {
    outputLinkNode.href = "#";
    outputLinkNode.classList.add("hidden");
  }
}

async function refreshReport(jobId) {
  if (renderedReportJobId === jobId && renderedReport) {
    return renderedReport;
  }

  const response = await fetch(`/jobs/${jobId}/artifacts/output.json`);
  const body = await response.json();
  if (!response.ok) {
    throw new Error(body.detail || "Could not load extraction report.");
  }

  renderSummary(body);
  renderedReportJobId = jobId;
  renderedReport = body;
  return body;
}

function updateJobHeader(jobId, state, updatedAt) {
  const resolvedJobId = jobId || "No active job";
  const resolvedState = state || "idle";
  const resolvedUpdatedAt = updatedAt ? new Date(updatedAt).toLocaleString() : "-";

  jobIdNode.textContent = resolvedJobId;
  jobIdRailNode.textContent = resolvedJobId;
  jobStateNode.textContent = resolvedState;
  jobStateRailNode.textContent = resolvedState;
  jobUpdatedNode.textContent = resolvedUpdatedAt;
  jobUpdatedRailNode.textContent = resolvedUpdatedAt;

  jobStatePillNode.dataset.state = resolvedState;
  jobStateRailPillNode.dataset.state = resolvedState;
}

function renderHistory(history) {
  jobHistoryNode.innerHTML = "";
  if (!history || history.length === 0) {
    const item = document.createElement("li");
    item.textContent = "Waiting for a job to start.";
    jobHistoryNode.appendChild(item);
    return;
  }

  for (const entry of history) {
    const item = document.createElement("li");
    const timestamp = new Date(entry.timestamp).toLocaleTimeString();
    item.textContent = `${entry.state} - ${entry.message} (${timestamp})`;
    jobHistoryNode.appendChild(item);
  }
}

function renderError(message) {
  if (!message) {
    jobErrorNode.textContent = "";
    jobErrorNode.classList.add("hidden");
    jobRunCardNode.classList.remove("rail-card--error");
    return;
  }

  jobErrorNode.textContent = message;
  jobErrorNode.classList.remove("hidden");
  jobRunCardNode.classList.add("rail-card--error");
}

function renderArtifacts(artifacts) {
  artifactListNode.innerHTML = "";
  artifactCountNode.textContent = String(artifacts.length);

  if (artifacts.length === 0) {
    const emptyItem = document.createElement("li");
    emptyItem.textContent = "No artifacts written yet.";
    artifactListNode.appendChild(emptyItem);
    return;
  }

  for (const artifact of artifacts) {
    artifactListNode.appendChild(makeArtifactListItem(artifact.name, artifact.url, artifact.size_bytes));
  }
}

function renderSummary(report) {
  reportSchemaNode.textContent = report.schema || "-";
  spacesCountNode.textContent = String(report.summary?.number_of_spaces ?? 0);
  openingsCountNode.textContent = String(report.summary?.number_of_openings ?? 0);
  missingCountNode.textContent = String(report.geometry_sanity?.spaces_with_missing_representation?.length ?? 0);
  invalidCountNode.textContent = String(report.geometry_sanity?.invalid_solids?.length ?? 0);
  workerBackendNode.textContent = report.preprocessing?.worker_backend || "-";
  validNormalizedCountNode.textContent = String(report.preprocessing?.summary?.valid_entities ?? 0);
  invalidNormalizedCountNode.textContent = String(report.preprocessing?.summary?.invalid_entities ?? 0);
  geometryUnitNode.textContent = report.preprocessing?.unit || "-";

  const rawMeshCount = countArtifacts(report.spaces || [], "raw_obj") + countArtifacts(report.openings || [], "raw_obj");
  const normalizedSpaceCount = countArtifacts(report.spaces || [], "normalized_obj");
  const normalizedOpeningCount = countArtifacts(report.openings || [], "normalized_obj");
  const failedCount = report.preprocessing?.summary?.invalid_entities ?? 0;

  layerRawCountNode.textContent = rawMeshCount > 0 ? String(rawMeshCount) : "n/a";
  layerNormalizedCountNode.textContent = String(normalizedSpaceCount);
  layerOpeningsCountNode.textContent = String(normalizedOpeningCount);
  layerFailedCountNode.textContent = String(failedCount);

  spacesObjectCountNode.textContent = String((report.spaces || []).filter((entity) => entity.geometry_ok).length);
  openingsObjectCountNode.textContent = String((report.openings || []).filter((entity) => entity.geometry_ok).length);
  failedObjectCountNode.textContent = String(failedCount);
}

function resetSummary() {
  reportSchemaNode.textContent = "-";
  spacesCountNode.textContent = "0";
  openingsCountNode.textContent = "0";
  missingCountNode.textContent = "0";
  invalidCountNode.textContent = "0";
  workerBackendNode.textContent = "-";
  validNormalizedCountNode.textContent = "0";
  invalidNormalizedCountNode.textContent = "0";
  geometryUnitNode.textContent = "-";

  layerRawCountNode.textContent = "n/a";
  layerNormalizedCountNode.textContent = "0";
  layerOpeningsCountNode.textContent = "0";
  layerFailedCountNode.textContent = "0";

  spacesObjectCountNode.textContent = "0";
  openingsObjectCountNode.textContent = "0";
  failedObjectCountNode.textContent = "0";
}

function hideQuickLinks() {
  statusLinkNode.href = "#";
  outputLinkNode.href = "#";
  viewerManifestLinkNode.href = "#";
  statusLinkNode.classList.add("hidden");
  outputLinkNode.classList.add("hidden");
  viewerManifestLinkNode.classList.add("hidden");
}

function makeArtifactListItem(label, url, sizeBytes) {
  const item = document.createElement("li");
  const link = document.createElement("a");
  link.href = url;
  link.textContent = label;
  link.target = "_blank";
  link.rel = "noreferrer";

  const size = document.createElement("span");
  size.textContent = sizeBytes == null ? "download" : formatBytes(sizeBytes);

  item.append(link, size);
  return item;
}

function countArtifacts(entities, artifactKey) {
  return entities.filter((entity) => Boolean(entity.artifacts?.[artifactKey])).length;
}

function setFlash(message, isError) {
  if (!message) {
    flashMessage.textContent = "";
    flashMessage.classList.add("hidden");
    delete flashMessage.dataset.state;
    return;
  }

  flashMessage.textContent = message;
  flashMessage.dataset.state = isError ? "error" : "info";
  flashMessage.classList.remove("hidden");
}

function formatBytes(size) {
  if (typeof size !== "number") {
    return "pending";
  }
  if (size < 1024) {
    return `${size} B`;
  }
  return `${(size / 1024).toFixed(1)} KB`;
}
