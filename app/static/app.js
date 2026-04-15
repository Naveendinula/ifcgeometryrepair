import { DebugViewer } from "./viewer.js?v=20260415-3";

const escapeHtml = globalThis.escapeHtml || ((value) =>
  String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;"));

if (globalThis.escapeHtml !== escapeHtml) {
  globalThis.escapeHtml = escapeHtml;
}

const form = document.querySelector("#upload-form");
const fileInput = document.querySelector("#file-input");
const externalShellModeInput = document.querySelector("#external-shell-mode-input");
const internalBoundaryThresholdInput = document.querySelector("#internal-boundary-threshold-input");
const alphaWrapAlphaInput = document.querySelector("#alpha-wrap-alpha-input");
const alphaWrapOffsetInput = document.querySelector("#alpha-wrap-offset-input");
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

const inspectSelectionNameNode = document.querySelector("#inspect-selection-name");
const inspectSelectionMetaNode = document.querySelector("#inspect-selection-meta");
const inspectSelectionActionsNode = document.querySelector("#inspect-selection-actions");
const jobHistoryOverviewNode = document.querySelector("#job-history-overview");
const jobHistoryDataNode = document.querySelector("#job-history-data");

const statusLinkNodes = [document.querySelector("#status-link"), document.querySelector("#status-link-rail")];
const outputLinkNodes = [document.querySelector("#output-link"), document.querySelector("#output-link-rail")];
const viewerManifestLinkNodes = [
  document.querySelector("#viewer-manifest-link"),
  document.querySelector("#viewer-manifest-link-rail"),
];

const pendingRemovalCountNode = document.querySelector("#pending-removal-count");
const pendingRemovalEmptyNode = document.querySelector("#pending-removal-empty");
const pendingRemovalListNode = document.querySelector("#pending-removal-list");
const clearPendingRemovalsButton = document.querySelector("#clear-pending-removals-button");
const rerunRemoveSpacesButton = document.querySelector("#rerun-remove-spaces-button");
const jobDerivationNoteNode = document.querySelector("#job-derivation-note");
const clashGroupCountNode = document.querySelector("#clash-group-count");
const clashReviewSummaryNode = document.querySelector("#clash-review-summary");
const clashReviewEmptyNode = document.querySelector("#clash-review-empty");
const clashGroupListNode = document.querySelector("#clash-group-list");
const acceptClashRecommendationsButton = document.querySelector("#accept-clash-recommendations-button");
const clearClashSelectionsButton = document.querySelector("#clear-clash-selections-button");
const rerunResolveClashesButton = document.querySelector("#rerun-resolve-clashes-button");

const overviewSpacesCountNode = document.querySelector("#overview-spaces-count");
const overviewOpeningsCountNode = document.querySelector("#overview-openings-count");
const overviewFailedCountNode = document.querySelector("#overview-failed-count");
const overviewPreflightStatusNode = document.querySelector("#overview-preflight-status");
const overviewSurfaceCountNode = document.querySelector("#overview-surface-count");
const overviewShellModeNode = document.querySelector("#overview-shell-mode");

const reportSchemaNode = document.querySelector("#report-schema");
const missingCountNode = document.querySelector("#missing-count");
const invalidCountNode = document.querySelector("#invalid-count");
const workerBackendNode = document.querySelector("#worker-backend");
const validNormalizedCountNode = document.querySelector("#valid-normalized-count");
const invalidNormalizedCountNode = document.querySelector("#invalid-normalized-count");
const preflightBlockerCountNode = document.querySelector("#preflight-blocker-count");
const preflightWarningCountNode = document.querySelector("#preflight-warning-count");
const geometryUnitNode = document.querySelector("#geometry-unit");
const unclassifiedSurfaceCountNode = document.querySelector("#unclassified-surface-count");
const internalVoidCountNode = document.querySelector("#internal-void-count");
const shellBackendNode = document.querySelector("#shell-backend");
const alphaWrapAlphaEffectiveNode = document.querySelector("#alpha-wrap-alpha-effective");
const alphaWrapOffsetEffectiveNode = document.querySelector("#alpha-wrap-offset-effective");

const layerRawCountNode = document.querySelector("#layer-raw-count");
const layerGridCountNode = document.querySelector("#layer-grid-count");
const layerNormalizedCountNode = document.querySelector("#layer-normalized-count");
const layerOpeningsCountNode = document.querySelector("#layer-openings-count");
const layerFailedCountNode = document.querySelector("#layer-failed-count");
const layerShellCountNode = document.querySelector("#layer-shell-count");
const layerSurfacesCountNode = document.querySelector("#layer-surfaces-count");

const artifactCountNode = document.querySelector("#artifact-count");
const artifactListNode = document.querySelector("#artifact-list");

const railTabButtons = [...document.querySelectorAll("[data-rail-tab-button]")];
const railTabPanels = [...document.querySelectorAll("[data-rail-panel]")];
const browserSearchNode = document.querySelector("#entity-browser-search");
const browserFilterButtons = [...document.querySelectorAll("[data-browser-filter]")];
const objectBrowserCountNode = document.querySelector("#object-browser-count");
const browserFilterCountNodes = {
  all: document.querySelector("#browser-filter-count-all"),
  spaces: document.querySelector("#browser-filter-count-spaces"),
  openings: document.querySelector("#browser-filter-count-openings"),
  failed: document.querySelector("#browser-filter-count-failed"),
  surfaces: document.querySelector("#browser-filter-count-surfaces"),
  marked: document.querySelector("#browser-filter-count-marked"),
};

const disclosureNodes = {
  overviewHistory: {
    button: document.querySelector('[data-disclosure-button="overviewHistory"]'),
    panel: document.querySelector("#overview-history-panel"),
  },
  inspectLayers: {
    button: document.querySelector('[data-disclosure-button="inspectLayers"]'),
    panel: document.querySelector("#inspect-layers-panel"),
  },
  metrics: {
    button: document.querySelector('[data-disclosure-button="metrics"]'),
    panel: document.querySelector("#data-section-metrics"),
  },
  artifacts: {
    button: document.querySelector('[data-disclosure-button="artifacts"]'),
    panel: document.querySelector("#data-section-artifacts"),
  },
  dataHistory: {
    button: document.querySelector('[data-disclosure-button="dataHistory"]'),
    panel: document.querySelector("#data-section-history"),
  },
};

let pollHandle = null;
let renderedReportJobId = null;
let renderedReport = null;
let currentJobId = null;
let currentJobState = "idle";
let rerunInFlight = false;
let clashRerunInFlight = false;
let activeRailTab = "overview";
let entityBrowserQuery = "";
let entityBrowserFilter = "all";
const pendingRemovals = new Map();
const clashResolutionSelections = new Map();
let focusedClashGroupId = null;

const dataDisclosureState = {
  overviewHistory: false,
  inspectLayers: false,
  metrics: true,
  artifacts: false,
  dataHistory: false,
};

const terminalStates = new Set(["complete", "failed"]);

const viewer = new DebugViewer({
  panel: document.querySelector("#viewer-panel"),
  manifestLinks: viewerManifestLinkNodes,
  canvas: document.querySelector("#viewer-canvas"),
  emptyState: document.querySelector("#viewer-empty"),
  emptyStateMessage: document.querySelector("#viewer-empty-message"),
  inspectSelectionName: inspectSelectionNameNode,
  inspectSelectionMeta: inspectSelectionMetaNode,
  inspectSelectionActions: inspectSelectionActionsNode,
  browserList: document.querySelector("#viewer-browser-list"),
  toggleGrid: document.querySelectorAll('[data-layer-toggle="grid"]'),
  toggleRaw: document.querySelectorAll('[data-layer-toggle="raw"]'),
  toggleNormalized: document.querySelectorAll('[data-layer-toggle="normalized"]'),
  toggleOpenings: document.querySelectorAll('[data-layer-toggle="openings"]'),
  toggleFailed: document.querySelectorAll('[data-layer-toggle="failed"]'),
  toggleShell: document.querySelectorAll('[data-layer-toggle="shell"]'),
  toggleSurfaces: document.querySelectorAll('[data-layer-toggle="surfaces"]'),
  resetViewButtons: [document.querySelector("#reset-view-button")],
  fitViewButtons: [document.querySelector("#fit-view-button")],
  canTogglePendingRemoval: (entity) => canManageSpaceRemovals() && isRemovableSpaceEntity(entity),
  isPendingRemoval: (entity) => isRemovableSpaceEntity(entity) && pendingRemovals.has(spaceRemovalKey(entity)),
  onTogglePendingRemoval: (entity) => togglePendingRemoval(entity),
  getBrowserQuery: () => entityBrowserQuery,
  getBrowserFilter: () => entityBrowserFilter,
  onBrowserStats: (stats) => renderBrowserStats(stats),
  revealSelectionInBrowser: (entity) => revealEntityInBrowser(entity),
  onSelectionChange: (entity) => {
    if (entity) {
      setActiveRailTab("inspect");
    }
  },
  onError: (message) => setFlash(message, true),
});

fileInput.addEventListener("change", () => updateSelectedFileName());
emptyUploadButton.addEventListener("click", () => fileInput.click());
clearPendingRemovalsButton.addEventListener("click", () => clearPendingRemovals());
rerunRemoveSpacesButton.addEventListener("click", () => rerunWithoutMarkedSpaces());
acceptClashRecommendationsButton.addEventListener("click", () => acceptRecommendedClashResolutions());
clearClashSelectionsButton.addEventListener("click", () => clearClashSelections());
rerunResolveClashesButton.addEventListener("click", () => rerunWithResolvedClashes());

for (const button of railTabButtons) {
  button.addEventListener("click", () => setActiveRailTab(button.dataset.railTabButton));
}

browserSearchNode.addEventListener("input", (event) => {
  entityBrowserQuery = event.target.value.trim();
  viewer.refreshEntityBrowser();
});

for (const button of browserFilterButtons) {
  button.addEventListener("click", () => {
    entityBrowserFilter = button.dataset.browserFilter;
    renderBrowserFilterState();
    viewer.refreshEntityBrowser();
  });
}

for (const [key, disclosure] of Object.entries(disclosureNodes)) {
  disclosure.button?.addEventListener("click", () => {
    dataDisclosureState[key] = !dataDisclosureState[key];
    renderDisclosureState();
  });
}

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
  payload.append("external_shell_mode", externalShellModeInput.value || "alpha_wrap");
  if (internalBoundaryThresholdInput?.value) {
    payload.append("internal_boundary_thickness_threshold_m", internalBoundaryThresholdInput.value);
  }
  if (alphaWrapAlphaInput?.value) {
    payload.append("alpha_wrap_alpha_m", alphaWrapAlphaInput.value);
  }
  if (alphaWrapOffsetInput?.value) {
    payload.append("alpha_wrap_offset_m", alphaWrapOffsetInput.value);
  }

  try {
    const response = await fetch("/jobs", {
      method: "POST",
      body: payload,
    });
    const body = await response.json();
    if (!response.ok) {
      throw new Error(body.detail || "Job creation failed.");
    }

    revealLinks(statusLinkNodes, body.status_url);
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
  resetClashReviewState();
  renderDerivation(null);
  resetRailState();
  clearPendingRemovals({ refreshViewer: false });
  hideQuickLinks();
  viewer.clear("Use the import control above to create a job and populate the 3D workspace.");
  setFlash("", false);
}

function clearWorkspaceForNewRun() {
  renderHistory([]);
  renderError("");
  renderArtifacts([]);
  resetSummary();
  resetClashReviewState();
  renderDerivation(null);
  resetRailState();
  clearPendingRemovals({ refreshViewer: false });
  hideQuickLinks();
  viewer.clear("Viewer layers will appear once processing completes.");
}

function resetRailState() {
  activeRailTab = "overview";
  entityBrowserQuery = "";
  entityBrowserFilter = "all";
  dataDisclosureState.overviewHistory = false;
  dataDisclosureState.inspectLayers = false;
  dataDisclosureState.metrics = true;
  dataDisclosureState.artifacts = false;
  dataDisclosureState.dataHistory = false;
  browserSearchNode.value = "";
  renderRailTabs();
  renderBrowserFilterState();
  renderDisclosureState();
  renderBrowserStats(emptyBrowserStats());
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
  renderDerivation(body.derivation || null);
  renderHistory(body.history);
  renderError(body.error);
  await refreshArtifacts(jobId);
  syncRerunControls();
  viewer.refreshRemovalControls();

  if (body.state === "complete" || body.state === "failed") {
    const report = await refreshReport(jobId);
    renderDerivation(report.derivation || body.derivation || null);
    await viewer.loadJob(jobId, report);
    viewer.refreshRemovalControls();
    setActiveRailTab("overview");
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
  revealLinks(statusLinkNodes, `/jobs/${jobId}`);

  if (artifactNames.has("output.json")) {
    revealLinks(outputLinkNodes, `/jobs/${jobId}/artifacts/output.json`);
  } else {
    hideLinkGroup(outputLinkNodes);
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

  renderedReportJobId = jobId;
  renderedReport = body;
  renderSummary(body);
  renderClashReview(body);
  return body;
}

function updateJobHeader(jobId, state, updatedAt) {
  const resolvedJobId = jobId || "No active job";
  const resolvedState = state || "idle";
  const resolvedUpdatedAt = updatedAt ? new Date(updatedAt).toLocaleString() : "-";
  currentJobId = jobId;
  currentJobState = resolvedState;

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
  const historyNodes = [jobHistoryOverviewNode, jobHistoryDataNode];
  for (const node of historyNodes) {
    node.innerHTML = "";
    if (!history || history.length === 0) {
      const item = document.createElement("li");
      item.textContent = "Waiting for a job to start.";
      node.appendChild(item);
      continue;
    }

    for (const entry of history) {
      const item = document.createElement("li");
      const timestamp = new Date(entry.timestamp).toLocaleTimeString();
      item.textContent = `${entry.state} - ${entry.message} (${timestamp})`;
      node.appendChild(item);
    }
  }
}

function renderError(message) {
  if (!message) {
    jobErrorNode.textContent = "";
    jobErrorNode.classList.add("hidden");
    jobRunCardNode.classList.remove("rail-hero--error");
    return;
  }

  jobErrorNode.textContent = message;
  jobErrorNode.classList.remove("hidden");
  jobRunCardNode.classList.add("rail-hero--error");
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
  const spacesCount = report.summary?.number_of_spaces ?? 0;
  const openingsCount = report.summary?.number_of_openings ?? 0;
  const missingCount = report.geometry_sanity?.spaces_with_missing_representation?.length ?? 0;
  const invalidCount = report.geometry_sanity?.invalid_solids?.length ?? 0;
  const validEntities = report.preprocessing?.summary?.valid_entities ?? 0;
  const invalidEntities = report.preprocessing?.summary?.invalid_entities ?? 0;
  const preflightStatus = report.preflight?.status || "-";
  const preflightBlockerCount = report.preflight?.summary?.blocker_count ?? 0;
  const preflightWarningCount = report.preflight?.summary?.warning_count ?? 0;
  const shellMode = report.external_shell?.mode_effective || "-";
  const surfaceCount = report.external_shell?.summary?.candidate_surface_count ?? 0;
  const unclassifiedCount = report.external_shell?.summary?.unclassified_count ?? 0;
  const internalVoidCount = report.external_shell?.summary?.per_class_counts?.internal_void ?? 0;
  const shellBackend = report.external_shell?.shell_backend || report.external_shell?.alpha_wrap?.backend || "-";
  const effectiveAlpha = report.external_shell?.alpha_wrap?.alpha_m_effective;
  const effectiveOffset = report.external_shell?.alpha_wrap?.offset_m_effective;

  const failedCount = [...(report.spaces || []), ...(report.openings || [])].filter(
    (entity) => entity.preflight_failed || !entity.geometry_ok,
  ).length;

  overviewSpacesCountNode.textContent = String(spacesCount);
  overviewOpeningsCountNode.textContent = String(openingsCount);
  overviewFailedCountNode.textContent = String(failedCount);
  overviewPreflightStatusNode.textContent = preflightStatus;
  overviewSurfaceCountNode.textContent = String(surfaceCount);
  overviewShellModeNode.textContent = shellMode;

  reportSchemaNode.textContent = report.schema || "-";
  missingCountNode.textContent = String(missingCount);
  invalidCountNode.textContent = String(invalidCount);
  workerBackendNode.textContent = report.preprocessing?.worker_backend || "-";
  validNormalizedCountNode.textContent = String(validEntities);
  invalidNormalizedCountNode.textContent = String(invalidEntities);
  preflightBlockerCountNode.textContent = String(preflightBlockerCount);
  preflightWarningCountNode.textContent = String(preflightWarningCount);
  geometryUnitNode.textContent = report.preprocessing?.unit || "-";
  unclassifiedSurfaceCountNode.textContent = String(unclassifiedCount);
  internalVoidCountNode.textContent = String(internalVoidCount);
  shellBackendNode.textContent = shellBackend;
  alphaWrapAlphaEffectiveNode.textContent = effectiveAlpha == null ? "-" : Number(effectiveAlpha).toFixed(3);
  alphaWrapOffsetEffectiveNode.textContent = effectiveOffset == null ? "-" : Number(effectiveOffset).toFixed(3);

  const rawMeshCount = countArtifacts(report.spaces || [], "raw_obj") + countArtifacts(report.openings || [], "raw_obj");
  const normalizedSpaceCount = countArtifacts(report.spaces || [], "normalized_obj");
  const normalizedOpeningCount = countArtifacts(report.openings || [], "normalized_obj");
  const gridCount = report.preprocessing?.artifacts?.viewer_manifest ? 1 : 0;
  const shellCount = report.external_shell?.artifacts?.shell_obj ? 1 : 0;

  layerRawCountNode.textContent = rawMeshCount > 0 ? String(rawMeshCount) : "n/a";
  layerGridCountNode.textContent = String(gridCount);
  layerNormalizedCountNode.textContent = String(normalizedSpaceCount);
  layerOpeningsCountNode.textContent = String(normalizedOpeningCount);
  layerFailedCountNode.textContent = String(failedCount);
  layerShellCountNode.textContent = String(shellCount);
  layerSurfacesCountNode.textContent = String(surfaceCount);
}

function resetSummary() {
  overviewSpacesCountNode.textContent = "0";
  overviewOpeningsCountNode.textContent = "0";
  overviewFailedCountNode.textContent = "0";
  overviewPreflightStatusNode.textContent = "-";
  overviewSurfaceCountNode.textContent = "0";
  overviewShellModeNode.textContent = "-";

  reportSchemaNode.textContent = "-";
  missingCountNode.textContent = "0";
  invalidCountNode.textContent = "0";
  workerBackendNode.textContent = "-";
  validNormalizedCountNode.textContent = "0";
  invalidNormalizedCountNode.textContent = "0";
  preflightBlockerCountNode.textContent = "0";
  preflightWarningCountNode.textContent = "0";
  geometryUnitNode.textContent = "-";
  unclassifiedSurfaceCountNode.textContent = "0";
  internalVoidCountNode.textContent = "0";
  shellBackendNode.textContent = "-";
  alphaWrapAlphaEffectiveNode.textContent = "-";
  alphaWrapOffsetEffectiveNode.textContent = "-";

  layerRawCountNode.textContent = "n/a";
  layerGridCountNode.textContent = "0";
  layerNormalizedCountNode.textContent = "0";
  layerOpeningsCountNode.textContent = "0";
  layerFailedCountNode.textContent = "0";
  layerShellCountNode.textContent = "0";
  layerSurfacesCountNode.textContent = "0";

  renderBrowserStats(emptyBrowserStats());
}

function resetClashReviewState() {
  clashResolutionSelections.clear();
  focusedClashGroupId = null;
  clashRerunInFlight = false;
  clashGroupCountNode.textContent = "0";
  clashReviewSummaryNode.textContent = "No clash review is required for the current job.";
  clashReviewEmptyNode.classList.remove("hidden");
  clashGroupListNode.innerHTML = "";
  clashGroupListNode.classList.add("hidden");
  acceptClashRecommendationsButton.disabled = true;
  clearClashSelectionsButton.disabled = true;
  rerunResolveClashesButton.disabled = true;
  rerunResolveClashesButton.textContent = "Create Clash Rerun";
  viewer.setHighlightedKeys([]);
}

function currentClashGroups() {
  return renderedReport?.preflight?.clash_groups || [];
}

function canManageClashReview() {
  return Boolean(currentJobId) && terminalStates.has(currentJobState) && currentClashGroups().length > 0;
}

function clashSelectionKey(spaceRef) {
  return spaceRef.global_id ? `global:${spaceRef.global_id}` : `express:${spaceRef.express_id}`;
}

function normalizeClashSpaceRef(spaceRef) {
  return {
    global_id: spaceRef.global_id || null,
    express_id: spaceRef.express_id,
    name: spaceRef.name || spaceRef.global_id || `#${spaceRef.express_id}`,
  };
}

function entityObjectNameFromRef(spaceRef) {
  return spaceRef?.global_id || `entity_${spaceRef?.express_id}`;
}

function getClashSelectionMap(groupId) {
  return clashResolutionSelections.get(groupId) || new Map();
}

function setClashSelection(groupId, spaceRefs) {
  const nextSelection = new Map();
  for (const spaceRef of spaceRefs) {
    const normalized = normalizeClashSpaceRef(spaceRef);
    nextSelection.set(clashSelectionKey(normalized), normalized);
  }
  if (nextSelection.size === 0) {
    clashResolutionSelections.delete(groupId);
  } else {
    clashResolutionSelections.set(groupId, nextSelection);
  }
}

function clearClashSelections() {
  clashResolutionSelections.clear();
  renderClashReview(renderedReport);
}

function acceptRecommendedClashResolutions() {
  if (!renderedReport) return;
  for (const clashGroup of currentClashGroups()) {
    const recommended = clashGroup.recommended_resolution;
    if (!recommended) continue;
    setClashSelection(clashGroup.clash_group_id, recommended.spaces_to_remove || []);
  }
  renderClashReview(renderedReport);
}

function toggleClashSpaceRemoval(groupId, spaceRef) {
  if (clashRerunInFlight) return;
  const nextSelection = new Map(getClashSelectionMap(groupId));
  const normalized = normalizeClashSpaceRef(spaceRef);
  const key = clashSelectionKey(normalized);
  if (nextSelection.has(key)) {
    nextSelection.delete(key);
  } else {
    nextSelection.set(key, normalized);
  }
  if (nextSelection.size === 0) {
    clashResolutionSelections.delete(groupId);
  } else {
    clashResolutionSelections.set(groupId, nextSelection);
  }
  renderClashReview(renderedReport);
}

function applyRecommendedClashResolution(groupId) {
  const clashGroup = currentClashGroups().find((candidate) => candidate.clash_group_id === groupId);
  if (!clashGroup?.recommended_resolution) return;
  setClashSelection(groupId, clashGroup.recommended_resolution.spaces_to_remove || []);
  renderClashReview(renderedReport);
}

function clearClashGroupSelection(groupId) {
  clashResolutionSelections.delete(groupId);
  renderClashReview(renderedReport);
}

function focusClashGroup(groupId) {
  focusedClashGroupId = focusedClashGroupId === groupId ? null : groupId;
  syncClashHighlights();
  renderClashReview(renderedReport);
}

function syncClashHighlights() {
  const clashGroup = currentClashGroups().find((candidate) => candidate.clash_group_id === focusedClashGroupId) || null;
  const keys = clashGroup ? clashGroup.spaces.map((spaceRef) => entityObjectNameFromRef(spaceRef)) : [];
  viewer.setHighlightedKeys(keys);
}

function buildClashResolutionPayload(clashGroups = currentClashGroups()) {
  const groupResolutions = [];
  for (const clashGroup of clashGroups) {
    const groupSelection = clashResolutionSelections.get(clashGroup.clash_group_id);
    if (!groupSelection || groupSelection.size === 0) continue;
    const selectedSpaces = [...groupSelection.values()];
    groupResolutions.push({
      clash_group_id: clashGroup.clash_group_id,
      remove_space_global_ids: selectedSpaces.filter((spaceRef) => Boolean(spaceRef.global_id)).map((spaceRef) => spaceRef.global_id),
      remove_space_express_ids: selectedSpaces.filter((spaceRef) => !spaceRef.global_id).map((spaceRef) => spaceRef.express_id),
    });
  }
  return { group_resolutions: groupResolutions };
}

function renderClashReview(report) {
  const clashGroups = report?.preflight?.clash_groups || [];
  clashGroupCountNode.textContent = String(clashGroups.length);

  if (clashGroups.length === 0) {
    resetClashReviewState();
    return;
  }

  if (!clashGroups.some((clashGroup) => clashGroup.clash_group_id === focusedClashGroupId)) {
    focusedClashGroupId = null;
    syncClashHighlights();
  }

  const recommendedGroupCount = clashGroups.filter((clashGroup) => clashGroup.recommended_resolution).length;
  const manualGroupCount = clashGroups.length - recommendedGroupCount;
  const resolutionPayload = buildClashResolutionPayload(clashGroups);
  const selectedSpaceCount = resolutionPayload.group_resolutions.reduce(
    (count, groupResolution) => count + groupResolution.remove_space_global_ids.length + groupResolution.remove_space_express_ids.length,
    0,
  );

  clashReviewSummaryNode.textContent =
    `${clashGroups.length} clash groups detected. ${recommendedGroupCount} have safe delete recommendations, ${manualGroupCount} require manual review.`;
  clashReviewEmptyNode.classList.add("hidden");
  clashGroupListNode.classList.remove("hidden");
  acceptClashRecommendationsButton.disabled = !canManageClashReview() || recommendedGroupCount === 0 || clashRerunInFlight;
  clearClashSelectionsButton.disabled = resolutionPayload.group_resolutions.length === 0 || clashRerunInFlight;
  rerunResolveClashesButton.disabled = !canManageClashReview() || resolutionPayload.group_resolutions.length === 0 || clashRerunInFlight;
  rerunResolveClashesButton.textContent = `Create Clash Rerun (${selectedSpaceCount} spaces)`;

  clashGroupListNode.innerHTML = "";
  for (const clashGroup of clashGroups) {
    const groupId = clashGroup.clash_group_id;
    const groupSelection = getClashSelectionMap(groupId);
    const card = document.createElement("article");
    card.className = "clash-group-card";
    card.dataset.state = focusedClashGroupId === groupId ? "focused" : "idle";
    card.innerHTML = `
      <div class="clash-group-card__header">
        <div>
          <strong>${escapeHtml(groupId)}</strong>
          <p>${escapeHtml(formatClashClassification(clashGroup.classification))}</p>
        </div>
        <span class="object-badge" data-kind="${escapeHtml(clashGroup.recommended_resolution ? "accent" : "danger")}">${escapeHtml(formatClashResolutionStatus(clashGroup.resolution_status))}</span>
      </div>
      <p class="clash-group-card__summary">${escapeHtml(buildClashEvidenceSummary(clashGroup))}</p>
    `;

    const actionRow = document.createElement("div");
    actionRow.className = "selection-actions selection-actions--inline";

    const focusButton = document.createElement("button");
    focusButton.type = "button";
    focusButton.className = "button button--ghost button--slate";
    focusButton.disabled = clashRerunInFlight;
    focusButton.textContent = focusedClashGroupId === groupId ? "Clear Focus" : "Focus Group";
    focusButton.addEventListener("click", () => focusClashGroup(groupId));
    actionRow.appendChild(focusButton);

    if (clashGroup.recommended_resolution) {
      const recommendationButton = document.createElement("button");
      recommendationButton.type = "button";
      recommendationButton.className = "button button--ghost button--slate";
      recommendationButton.disabled = clashRerunInFlight;
      recommendationButton.textContent = "Use Recommendation";
      recommendationButton.addEventListener("click", () => applyRecommendedClashResolution(groupId));
      actionRow.appendChild(recommendationButton);
    }

    const clearButton = document.createElement("button");
    clearButton.type = "button";
    clearButton.className = "button button--ghost button--slate";
    clearButton.disabled = clashRerunInFlight || groupSelection.size === 0;
    clearButton.textContent = "Clear Group";
    clearButton.addEventListener("click", () => clearClashGroupSelection(groupId));
    actionRow.appendChild(clearButton);

    card.appendChild(actionRow);

    const spaceList = document.createElement("div");
    spaceList.className = "clash-space-list";
    for (const spaceRef of clashGroup.spaces || []) {
      const selectionKey = clashSelectionKey(spaceRef);
      const isSelected = groupSelection.has(selectionKey);
      const row = document.createElement("div");
      row.className = "clash-space-row";
      row.dataset.state = isSelected ? "remove" : "keep";

      const info = document.createElement("div");
      info.className = "clash-space-row__info";
      info.innerHTML = `
        <strong>${escapeHtml(spaceRef.name || spaceRef.global_id || `#${spaceRef.express_id}`)}</strong>
        <small>${escapeHtml((spaceRef.global_id || `#${spaceRef.express_id}`) + (spaceRef.recommended_action ? ` | recommended ${spaceRef.recommended_action}` : ""))}</small>
      `;

      const rowActions = document.createElement("div");
      rowActions.className = "clash-space-row__actions";

      const locateButton = document.createElement("button");
      locateButton.type = "button";
      locateButton.className = "button button--ghost button--slate";
      locateButton.disabled = clashRerunInFlight;
      locateButton.textContent = "Locate";
      locateButton.addEventListener("click", () => {
        focusedClashGroupId = groupId;
        syncClashHighlights();
        renderClashReview(renderedReport);
        viewer.selectEntity(entityObjectNameFromRef(spaceRef), { scrollIntoView: true, revealSelection: true });
      });
      rowActions.appendChild(locateButton);

      const toggleButton = document.createElement("button");
      toggleButton.type = "button";
      toggleButton.className = "button button--ghost button--slate";
      toggleButton.disabled = clashRerunInFlight;
      toggleButton.textContent = isSelected ? "Keep" : "Remove";
      toggleButton.addEventListener("click", () => toggleClashSpaceRemoval(groupId, spaceRef));
      rowActions.appendChild(toggleButton);

      row.append(info, rowActions);
      spaceList.appendChild(row);
    }
    card.appendChild(spaceList);

    const footer = document.createElement("p");
    footer.className = "clash-group-card__footer";
    footer.textContent =
      groupSelection.size > 0
        ? `${groupSelection.size} spaces selected for removal in this group.`
        : clashGroup.recommended_resolution
          ? "No removals selected yet. Use the recommendation or override it manually."
          : "No safe automatic recommendation is available for this group.";
    card.appendChild(footer);

    clashGroupListNode.appendChild(card);
  }
}

function renderDerivation(derivation) {
  if (!derivation) {
    jobDerivationNoteNode.textContent = "Original upload. Mark spaces to create a filtered rerun.";
    jobDerivationNoteNode.dataset.state = "base";
    return;
  }

  if (derivation.operation === "resolve_space_clashes") {
    jobDerivationNoteNode.textContent =
      `Derived from ${derivation.parent_job_id}, reviewed ${derivation.resolved_clash_group_count || 0} clash groups, and removed ${derivation.removed_space_count} spaces. Root job ${derivation.root_job_id}.`;
  } else {
    jobDerivationNoteNode.textContent =
      `Derived from ${derivation.parent_job_id}, removed ${derivation.removed_space_count} spaces. Root job ${derivation.root_job_id}.`;
  }
  jobDerivationNoteNode.dataset.state = "derived";
}

function hideQuickLinks() {
  hideLinkGroup(statusLinkNodes);
  hideLinkGroup(outputLinkNodes);
  hideLinkGroup(viewerManifestLinkNodes);
}

function revealLinks(nodes, href) {
  for (const node of nodes) {
    node.href = href;
    node.classList.remove("hidden");
  }
}

function hideLinkGroup(nodes) {
  for (const node of nodes) {
    node.href = "#";
    node.classList.add("hidden");
  }
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

function canManageSpaceRemovals() {
  return Boolean(currentJobId) && terminalStates.has(currentJobState);
}

function isRemovableSpaceEntity(entity) {
  return Boolean(entity && entity.selection_type === "entity" && entity.entity_type === "IfcSpace");
}

function spaceRemovalKey(entity) {
  return entity.global_id ? `global:${entity.global_id}` : `express:${entity.express_id}`;
}

function normalizeRemovalEntry(entity) {
  return {
    global_id: entity.global_id || null,
    express_id: entity.express_id,
    name: entity.name || entity.object_name,
    storey: entity.storey || null,
  };
}

function togglePendingRemoval(entity) {
  if (!canManageSpaceRemovals() || !isRemovableSpaceEntity(entity)) {
    return;
  }

  const key = spaceRemovalKey(entity);
  if (pendingRemovals.has(key)) {
    pendingRemovals.delete(key);
  } else {
    pendingRemovals.set(key, normalizeRemovalEntry(entity));
  }
  renderPendingRemovals();
  viewer.refreshRemovalControls();
}

function clearPendingRemovals({ refreshViewer = true } = {}) {
  pendingRemovals.clear();
  renderPendingRemovals();
  if (refreshViewer) {
    viewer.refreshRemovalControls();
  }
}

function renderPendingRemovals() {
  const pendingItems = [...pendingRemovals.values()].sort(comparePendingRemovalEntries);
  const count = pendingItems.length;
  pendingRemovalCountNode.textContent = String(count);
  rerunRemoveSpacesButton.textContent = `Rerun Without ${count} Spaces`;
  rerunRemoveSpacesButton.disabled = !canManageSpaceRemovals() || count === 0 || rerunInFlight;
  clearPendingRemovalsButton.disabled = count === 0 || rerunInFlight;

  pendingRemovalListNode.innerHTML = "";
  if (count === 0) {
    pendingRemovalEmptyNode.classList.remove("hidden");
    pendingRemovalListNode.classList.add("hidden");
    return;
  }

  pendingRemovalEmptyNode.classList.add("hidden");
  pendingRemovalListNode.classList.remove("hidden");
  for (const entry of pendingItems) {
    const item = document.createElement("li");
    const label = document.createElement("span");
    const storeyName = entry.storey?.name ? ` | ${entry.storey.name}` : "";
    const identifier = entry.global_id || `#${entry.express_id}`;
    label.textContent = `${entry.name || identifier}${storeyName} (${identifier})`;
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = "Unmark";
    button.disabled = rerunInFlight;
    button.addEventListener("click", () => {
      pendingRemovals.delete(spaceRemovalKey(entry));
      renderPendingRemovals();
      viewer.refreshRemovalControls();
    });
    item.append(label, button);
    pendingRemovalListNode.appendChild(item);
  }
}

function syncRerunControls() {
  renderPendingRemovals();
}

async function rerunWithoutMarkedSpaces() {
  if (!canManageSpaceRemovals() || pendingRemovals.size === 0 || !currentJobId) {
    return;
  }

  rerunInFlight = true;
  syncRerunControls();
  viewer.refreshRemovalControls();
  setFlash(`Creating derived rerun from job ${currentJobId}...`, false);

  const payload = {
    space_global_ids: [],
    space_express_ids: [],
  };
  for (const entry of pendingRemovals.values()) {
    if (entry.global_id) {
      payload.space_global_ids.push(entry.global_id);
    } else {
      payload.space_express_ids.push(entry.express_id);
    }
  }

  try {
    const response = await fetch(`/jobs/${currentJobId}/rerun/remove-spaces`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
    });
    const body = await response.json();
    if (!response.ok) {
      throw new Error(body.detail || "Could not create the derived rerun.");
    }

    stopPolling();
    renderedReportJobId = null;
    renderedReport = null;
    clearWorkspaceForNewRun();
    renderDerivation({
      parent_job_id: body.parent_job_id,
      root_job_id: body.root_job_id,
      removed_space_count: body.removed_space_count,
    });
    updateJobHeader(body.job_id, body.state, body.created_at);
    setFlash(`Derived job ${body.job_id} created from ${body.parent_job_id}.`, false);
    await refreshStatus(body.job_id);
    startPolling(body.job_id);
  } catch (error) {
    setFlash(error.message, true);
  } finally {
    rerunInFlight = false;
    syncRerunControls();
    viewer.refreshRemovalControls();
  }
}

async function rerunWithResolvedClashes() {
  const clashGroups = currentClashGroups();
  const payload = buildClashResolutionPayload(clashGroups);
  if (!canManageClashReview() || payload.group_resolutions.length === 0 || !currentJobId) {
    return;
  }

  clashRerunInFlight = true;
  renderClashReview(renderedReport);
  viewer.refreshRemovalControls();
  setFlash(`Creating clash-resolution rerun from job ${currentJobId}...`, false);

  try {
    const response = await fetch(`/jobs/${currentJobId}/rerun/resolve-space-clashes`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
    });
    const body = await response.json();
    if (!response.ok) {
      throw new Error(body.detail || "Could not create the clash-resolution rerun.");
    }

    stopPolling();
    renderedReportJobId = null;
    renderedReport = null;
    clearWorkspaceForNewRun();
    renderDerivation({
      parent_job_id: body.parent_job_id,
      root_job_id: body.root_job_id,
      operation: "resolve_space_clashes",
      removed_space_count: body.removed_space_count,
      resolved_clash_group_count: body.resolved_clash_group_count || payload.group_resolutions.length,
    });
    updateJobHeader(body.job_id, body.state, body.created_at);
    setFlash(`Derived clash-resolution job ${body.job_id} created from ${body.parent_job_id}.`, false);
    await refreshStatus(body.job_id);
    startPolling(body.job_id);
  } catch (error) {
    setFlash(error.message, true);
  } finally {
    clashRerunInFlight = false;
    renderClashReview(renderedReport);
    viewer.refreshRemovalControls();
  }
}

function setActiveRailTab(tab) {
  activeRailTab = tab;
  renderRailTabs();
}

function renderRailTabs() {
  for (const button of railTabButtons) {
    const isActive = button.dataset.railTabButton === activeRailTab;
    button.dataset.state = isActive ? "active" : "idle";
    button.setAttribute("aria-selected", String(isActive));
  }

  for (const panel of railTabPanels) {
    panel.classList.toggle("hidden", panel.dataset.railPanel !== activeRailTab);
  }
}

function renderBrowserFilterState() {
  for (const button of browserFilterButtons) {
    const isActive = button.dataset.browserFilter === entityBrowserFilter;
    button.dataset.state = isActive ? "active" : "idle";
    button.setAttribute("aria-pressed", String(isActive));
  }
}

function renderBrowserStats(stats) {
  objectBrowserCountNode.textContent = String(stats.all);
  browserFilterCountNodes.all.textContent = String(stats.all);
  browserFilterCountNodes.spaces.textContent = String(stats.spaces);
  browserFilterCountNodes.openings.textContent = String(stats.openings);
  browserFilterCountNodes.failed.textContent = String(stats.failed);
  browserFilterCountNodes.surfaces.textContent = String(stats.surfaces);
  browserFilterCountNodes.marked.textContent = String(stats.marked);
}

function renderDisclosureState() {
  for (const [key, disclosure] of Object.entries(disclosureNodes)) {
    const isOpen = Boolean(dataDisclosureState[key]);
    disclosure.button?.setAttribute("aria-expanded", String(isOpen));
    disclosure.panel?.classList.toggle("hidden", !isOpen);
  }
}

function revealEntityInBrowser(entity) {
  if (!entity) return false;
  let changed = false;
  if (!entityMatchesBrowserFilter(entity, entityBrowserFilter)) {
    entityBrowserFilter = "all";
    renderBrowserFilterState();
    changed = true;
  }
  if (!entityMatchesBrowserQuery(entity, entityBrowserQuery)) {
    entityBrowserQuery = "";
    browserSearchNode.value = "";
    changed = true;
  }
  return changed;
}

function emptyBrowserStats() {
  return {
    all: 0,
    spaces: 0,
    openings: 0,
    failed: 0,
    surfaces: 0,
    marked: 0,
  };
}

function formatClashClassification(value) {
  switch ((value || "").trim()) {
    case "exact_duplicate":
      return "Exact Duplicate";
    case "contained_fragment":
      return "Contained Fragment";
    case "partial_overlap":
      return "Partial Overlap";
    case "self_intersection_only":
      return "Self-Intersection";
    default:
      return value || "Clash Group";
  }
}

function formatClashResolutionStatus(value) {
  switch ((value || "").trim()) {
    case "recommended":
      return "Recommended";
    case "manual_required":
      return "Manual Review";
    case "clear":
      return "Clear";
    default:
      return value || "Review";
  }
}

function buildClashEvidenceSummary(clashGroup) {
  const pairCount = Array.isArray(clashGroup.pairs) ? clashGroup.pairs.length : 0;
  const spaceCount = Array.isArray(clashGroup.spaces) ? clashGroup.spaces.length : 0;
  const reason = clashGroup.recommended_resolution?.reason;
  if (reason) {
    return `${spaceCount} spaces, ${pairCount} overlap pairs. ${reason}`;
  }
  if (clashGroup.classification === "partial_overlap") {
    return `${spaceCount} spaces, ${pairCount} overlap pairs. No safe delete recommendation is available.`;
  }
  if (clashGroup.classification === "self_intersection_only") {
    return "This space self-intersects and has no automatic delete recommendation.";
  }
  return `${spaceCount} spaces, ${pairCount} overlap pairs. Review and choose removals before creating a rerun.`;
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

function comparePendingRemovalEntries(left, right) {
  return (left.name || "").localeCompare(right.name || "");
}

function entityMatchesBrowserFilter(entity, filterValue) {
  const filter = (filterValue || "all").trim().toLowerCase();
  if (filter === "all") return true;
  if (filter === "marked") return isRemovableSpaceEntity(entity) && pendingRemovals.has(spaceRemovalKey(entity));
  return browserEntityGroup(entity) === filter;
}

function entityMatchesBrowserQuery(entity, queryValue) {
  const query = (queryValue || "").trim().toLowerCase();
  if (!query) return true;
  const haystack = [
    entity.name,
    entity.object_name,
    entity.global_id,
    entity.surface_id,
    entity.entity_type,
    entity.classification,
    entity.storey?.name,
    entity.building?.name,
    entity.express_id == null ? null : String(entity.express_id),
    browserEntityBadgeLabel(entity),
  ].filter(Boolean).join(" ").toLowerCase();
  return haystack.includes(query);
}

function browserEntityGroup(entity) {
  if (entity.selection_type === "surface") return "surfaces";
  if (entity.failed) return "failed";
  if (entity.entity_type === "IfcSpace") return "spaces";
  if (entity.entity_type === "IfcOpeningElement") return "openings";
  return "other";
}

function browserEntityBadgeLabel(entity) {
  if (entity.selection_type === "surface") return entity.classification || "surface";
  if (entity.failed) return "failed";
  if (entity.entity_type === "IfcSpace") return entity.storey?.name || "space";
  if (entity.entity_type === "IfcOpeningElement") return "opening";
  return entity.entity_type || "entity";
}
