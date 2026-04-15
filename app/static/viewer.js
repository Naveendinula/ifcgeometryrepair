import {
  AmbientLight,
  Box3,
  Color,
  DirectionalLight,
  DoubleSide,
  EdgesGeometry,
  Group,
  LineBasicMaterial,
  LineSegments,
  Mesh,
  MeshBasicMaterial,
  PlaneGeometry,
  MeshStandardMaterial,
  PerspectiveCamera,
  Raycaster,
  Scene,
  ShaderMaterial,
  SphereGeometry,
  Vector2,
  Vector3,
  WebGLRenderer,
} from "three";
import { OrbitControls } from "/static/vendor/three/examples/jsm/controls/OrbitControls.js";
import { OBJLoader } from "/static/vendor/three/examples/jsm/loaders/OBJLoader.js";
import { GLTFLoader } from "/static/vendor/three/examples/jsm/loaders/GLTFLoader.js";

const escapeHtml = globalThis.escapeHtml || ((value) =>
  String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;"));

if (globalThis.escapeHtml !== escapeHtml) {
  globalThis.escapeHtml = escapeHtml;
}

const LAYERS = {
  grid: "gridFloor",
  raw: "raw",
  spaces: "normalizedSpaces",
  openings: "openings",
  failed: "failed",
  shell: "envelopeShell",
  surfaces: "surfaceClassification",
};

const EMPTY = {
  initial: "Use the import control above to create a job and populate the 3D workspace.",
  loading: "Loading viewer artifacts...",
  unavailable: "No viewer geometry is available for this job.",
  disabled: "Enable at least one layer to inspect the job geometry.",
  ready: "",
};

export class DebugViewer {
  constructor(options) {
    Object.assign(this, {
      panel: options.panel,
      canvas: options.canvas,
      emptyState: options.emptyState,
      emptyStateMessage: options.emptyStateMessage || options.emptyState,
      inspectSelectionName: options.inspectSelectionName,
      inspectSelectionMeta: options.inspectSelectionMeta,
      inspectSelectionActions: options.inspectSelectionActions,
      browserList: options.browserList,
      manifestLinks: normalizeElements(options.manifestLinks || options.manifestLink),
      canTogglePendingRemoval: options.canTogglePendingRemoval || null,
      isPendingRemoval: options.isPendingRemoval || null,
      onTogglePendingRemoval: options.onTogglePendingRemoval || null,
      getBrowserQuery: options.getBrowserQuery || (() => ""),
      getBrowserFilter: options.getBrowserFilter || (() => "all"),
      onBrowserStats: options.onBrowserStats || null,
      revealSelectionInBrowser: options.revealSelectionInBrowser || null,
      onSelectionChange: options.onSelectionChange || null,
      onError: options.onError || null,
      objLoader: new OBJLoader(),
      gltfLoader: new GLTFLoader(),
      raycaster: new Raycaster(),
      pointer: new Vector2(),
      currentJobId: null,
      manifest: null,
      selectedKey: null,
      entityMap: new Map(),
      listButtons: new Map(),
      listRows: new Map(),
      selectableMeshes: [],
      layerRoots: new Map(),
      focusRoots: new Map(),
      lastClickSelection: null,
      highlightedKeys: [],
      contentBounds: null,
      focusBounds: null,
      homeCameraFrame: null,
      gridFloorRoot: null,
      gridFloorMaterial: null,
      gridBaseSpan: 0,
    });

    this.layerObjectMaps = Object.fromEntries(Object.values(LAYERS).map((key) => [key, new Map()]));
    this.toggleGroups = {
      [LAYERS.grid]: normalizeElements(options.toggleGrid),
      [LAYERS.raw]: normalizeElements(options.toggleRaw),
      [LAYERS.spaces]: normalizeElements(options.toggleNormalized),
      [LAYERS.openings]: normalizeElements(options.toggleOpenings),
      [LAYERS.failed]: normalizeElements(options.toggleFailed),
      [LAYERS.shell]: normalizeElements(options.toggleShell),
      [LAYERS.surfaces]: normalizeElements(options.toggleSurfaces),
    };
    this.resetViewButtons = normalizeElements(options.resetViewButtons);
    this.fitViewButtons = normalizeElements(options.fitViewButtons);

    this.scene = new Scene();
    this.camera = new PerspectiveCamera(52, 1, 0.01, 100000);
    this.camera.up.set(0, 0, 1);
    this.camera.position.set(12, -10, 12);
    this.renderer = new WebGLRenderer({ canvas: this.canvas, antialias: true, alpha: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    this.controls = new OrbitControls(this.camera, this.canvas);
    this.controls.enableDamping = true;
    this.controls.screenSpacePanning = false;
    this.controls.target.set(0, 0, 0);

    this.scene.add(new AmbientLight(0xffffff, 1.35));
    const keyLight = new DirectionalLight(0xffffff, 1.15);
    keyLight.position.set(12, 20, 8);
    this.scene.add(keyLight);
    const fillLight = new DirectionalLight(0xe4edf5, 0.55);
    fillLight.position.set(-8, 10, -10);
    this.scene.add(fillLight);

    this.helperRoot = new Group();
    this.scene.add(this.helperRoot);
    this.modelRoot = new Group();
    this.scene.add(this.modelRoot);
    this.selectionOverlayRoot = new Group();
    this.scene.add(this.selectionOverlayRoot);

    for (const toggles of Object.values(this.toggleGroups)) {
      for (const toggle of toggles) {
        toggle.addEventListener("change", () => {
          this.syncToggleGroup(toggles, toggle.checked);
          this.applyLayerVisibility();
        });
      }
    }
    for (const button of this.resetViewButtons) button?.addEventListener("click", () => this.resetCamera());
    for (const button of this.fitViewButtons) button?.addEventListener("click", () => this.fitCamera());

    this.canvas.addEventListener("click", (event) => this.handleCanvasClick(event));
    window.addEventListener("resize", () => this.resize());
    if (typeof ResizeObserver !== "undefined") {
      this.resizeObserver = new ResizeObserver(() => this.resize());
      this.resizeObserver.observe(this.canvas.parentElement);
    }

    this.animate = this.animate.bind(this);
    window.requestAnimationFrame(this.animate);
    this.clear();
  }

  async loadJob(jobId, report) {
    const manifestPath = report?.preprocessing?.artifacts?.viewer_manifest;
    if (!manifestPath) {
      this.clear(EMPTY.unavailable);
      return;
    }
    if (this.currentJobId === jobId && this.manifest) {
      this.setSceneActionState(true);
      this.refreshEntityBrowser();
      this.resize();
      return;
    }

    this.clear(EMPTY.loading);
    const manifestUrl = `/jobs/${jobId}/artifacts/${manifestPath}`;
    const response = await fetch(manifestUrl);
    const manifest = await response.json();
    if (!response.ok) {
      this.clear(EMPTY.unavailable);
      throw new Error(manifest.detail || "Could not load viewer manifest.");
    }

    this.currentJobId = jobId;
    this.manifest = manifest;
    this.entityMap = new Map([...manifest.entities, ...(manifest.surface_entities || [])].map((e) => [e.object_name, e]));
    for (const link of this.manifestLinks) {
      link.href = manifestUrl;
      link.classList.remove("hidden");
    }

    await Promise.all([
      this.loadRawLayer(jobId, manifest.layers.raw_ifc_preview),
      this.loadAggregateLayer(jobId, LAYERS.spaces, manifest.layers.normalized_spaces),
      this.loadAggregateLayer(jobId, LAYERS.openings, manifest.layers.openings),
      this.loadShellLayer(jobId, manifest.layers.envelope_shell),
      this.loadAggregateLayer(jobId, LAYERS.surfaces, manifest.layers.surface_classification),
    ]);
    this.buildFailedLayer();
    this.contentBounds = this.computeContentBounds();
    this.focusBounds = this.computeFocusBounds() || this.contentBounds;
    this.homeCameraFrame = this.focusBounds ? buildCameraFrame(this.focusBounds) : null;
    this.buildGridFloor();
    this.configureToggles(manifest);
    this.setSceneActionState(true);
    this.refreshEntityBrowser();
    this.applyLayerVisibility();
    this.renderInspectSummary(null);
    this.resize();
    this.fitCamera();
  }

  clear(message = EMPTY.initial) {
    this.currentJobId = null;
    this.manifest = null;
    this.selectedKey = null;
    this.highlightedKeys = [];
    this.entityMap = new Map();
    this.listButtons = new Map();
    this.listRows = new Map();
    this.selectableMeshes = [];
    this.focusRoots = new Map();
    this.lastClickSelection = null;
    this.contentBounds = null;
    this.focusBounds = null;
    this.homeCameraFrame = null;
    this.gridFloorRoot = null;
    this.gridFloorMaterial = null;
    this.gridBaseSpan = 0;
    this.modelRoot.position.set(0, 0, 0);
    for (const link of this.manifestLinks) {
      link.href = "#";
      link.classList.add("hidden");
    }
    this.refreshEntityBrowser();
    this.renderInspectSummary(null);
    for (const layerId of Object.values(LAYERS)) this.disposeLayer(layerId);
    this.clearSelectionOverlay();
    for (const layerId of Object.values(LAYERS)) this.setToggleState(this.toggleGroups[layerId], false, false);
    this.setSceneActionState(false);
    this.renderEmptyState(message);
    this.resetCamera();
  }

  animate() {
    this.controls.update();
    this.updateGridFloorPlacement();
    this.renderer.render(this.scene, this.camera);
    window.requestAnimationFrame(this.animate);
  }

  syncToggleGroup(toggles, checked) {
    for (const toggle of toggles) toggle.checked = !toggle.disabled && checked;
  }

  setSceneActionState(enabled) {
    for (const button of [...this.resetViewButtons, ...this.fitViewButtons]) {
      if (button) button.disabled = !enabled;
    }
  }

  async loadRawLayer(jobId, layerConfig) {
    if (!layerConfig?.available) return;
    const root = new Group();
    let rawSpacesRoot = null;
    for (const [assetPath, group] of [
      [layerConfig.spaces_obj, "spaces"],
      [layerConfig.openings_obj, "openings"],
    ]) {
      if (!assetPath) continue;
      const object = await this.loadAsset(jobId, assetPath, null);
      root.add(object);
      this.indexNamedRoots(object, LAYERS.raw);
      if (group === "spaces") rawSpacesRoot = object;
    }
    this.layerRoots.set(LAYERS.raw, root);
    if (rawSpacesRoot) this.focusRoots.set(LAYERS.raw, rawSpacesRoot);
    this.modelRoot.add(root);
  }

  async loadAggregateLayer(jobId, layerId, layerConfig) {
    if (!layerConfig?.available) return;
    const object = await this.loadAsset(jobId, layerConfig.obj, layerConfig.glb);
    this.layerRoots.set(layerId, object);
    if (layerId === LAYERS.spaces) this.focusRoots.set(layerId, object);
    this.modelRoot.add(object);
    this.indexNamedRoots(object, layerId);
  }

  async loadShellLayer(jobId, layerConfig) {
    if (!layerConfig?.available) return;
    const object = await this.loadAsset(jobId, layerConfig.obj, layerConfig.glb);
    object.traverse((node) => {
      if (node.isMesh) node.material = this.createMaterial(LAYERS.shell, null);
    });
    this.layerRoots.set(LAYERS.shell, object);
    this.focusRoots.set(LAYERS.shell, object);
    this.modelRoot.add(object);
  }

  async loadAsset(jobId, objPath, glbPath) {
    const artifactPath = glbPath || objPath;
    if (!artifactPath) return new Group();
    const url = `/jobs/${jobId}/artifacts/${artifactPath}`;
    if (glbPath) {
      const gltf = await new Promise((resolve, reject) => this.gltfLoader.load(url, resolve, undefined, reject));
      return gltf.scene;
    }
    return new Promise((resolve, reject) => this.objLoader.load(url, resolve, undefined, reject));
  }

  indexNamedRoots(root, layerId) {
    for (const namedRoot of this.collectNamedRoots(root)) {
      const entity = this.entityMap.get(namedRoot.name);
      if (!entity) continue;
      namedRoot.userData.selectKey = namedRoot.name;
      namedRoot.userData.layerId = layerId;
      this.layerObjectMaps[layerId].set(namedRoot.name, namedRoot);
      this.applyStyle(namedRoot, layerId, entity);
      namedRoot.traverse((node) => {
        if (!node.isMesh) return;
        node.userData.selectKey = namedRoot.name;
        node.userData.layerId = layerId;
        this.selectableMeshes.push(node);
      });
    }
  }

  collectNamedRoots(root) {
    const result = [];
    const visit = (node, isRoot) => {
      if (!isRoot && this.entityMap.has(node.name)) {
        result.push(node);
        return;
      }
      for (const child of node.children) visit(child, false);
    };
    visit(root, true);
    return result;
  }

  applyStyle(root, layerId, entity) {
    root.traverse((node) => {
      if (node.isMesh) node.material = this.createMaterial(layerId, entity);
    });
  }

  createMaterial(layerId, entity) {
    if (layerId === LAYERS.raw) return new MeshBasicMaterial({ color: 0x95a3b2, transparent: true, opacity: 0.2, depthWrite: false, side: DoubleSide });
    if (layerId === LAYERS.shell) return new MeshBasicMaterial({ color: 0xff8a57, transparent: true, opacity: 0.16, depthWrite: false, side: DoubleSide });
    if (layerId === LAYERS.failed) return new MeshBasicMaterial({ color: 0xda4757, transparent: true, opacity: 0.58, depthWrite: false, wireframe: true, side: DoubleSide });
    if (layerId === LAYERS.openings) return new MeshStandardMaterial({ color: 0x3259d7, transparent: true, opacity: 0.9, roughness: 0.58, metalness: 0.0, side: DoubleSide });
    if (layerId === LAYERS.surfaces) return new MeshStandardMaterial({ color: colorForSurfaceClassification(entity?.classification), transparent: true, opacity: 0.86, roughness: 0.46, metalness: 0.0, side: DoubleSide });
    return new MeshStandardMaterial({ color: colorForEntity(entity?.object_name || "entity"), transparent: true, opacity: 0.92, roughness: 0.72, metalness: 0.02, side: DoubleSide });
  }

  buildFailedLayer() {
    this.disposeLayer(LAYERS.failed);
    if (!this.manifest) return;
    const root = new Group();
    for (const entity of this.manifest.entities.filter((item) => item.failed)) {
      const source = this.layerObjectMaps[LAYERS.spaces].get(entity.object_name) || this.layerObjectMaps[LAYERS.openings].get(entity.object_name) || this.layerObjectMaps[LAYERS.raw].get(entity.object_name);
      if (source) {
        const clone = source.clone(true);
        clone.name = entity.object_name;
        clone.userData.selectKey = entity.object_name;
        clone.userData.layerId = LAYERS.failed;
        this.applyStyle(clone, LAYERS.failed, entity);
        this.layerObjectMaps[LAYERS.failed].set(entity.object_name, clone);
        clone.traverse((node) => {
          if (!node.isMesh) return;
          node.userData.selectKey = entity.object_name;
          node.userData.layerId = LAYERS.failed;
          this.selectableMeshes.push(node);
        });
        root.add(clone);
      } else if (Array.isArray(entity.marker_origin)) {
        const marker = new Mesh(new SphereGeometry(0.15, 16, 16), new MeshBasicMaterial({ color: 0xda4757, transparent: true, opacity: 0.95 }));
        marker.position.set(...entity.marker_origin);
        marker.name = entity.object_name;
        marker.userData.selectKey = entity.object_name;
        marker.userData.layerId = LAYERS.failed;
        this.layerObjectMaps[LAYERS.failed].set(entity.object_name, marker);
        this.selectableMeshes.push(marker);
        root.add(marker);
      }
    }
    this.layerRoots.set(LAYERS.failed, root);
    this.modelRoot.add(root);
  }

  configureToggles(manifest) {
    this.setToggleState(this.toggleGroups[LAYERS.grid], Boolean(this.focusBounds || this.contentBounds), true);
    this.setToggleState(this.toggleGroups[LAYERS.raw], manifest.layers.raw_ifc_preview.available, false);
    this.setToggleState(this.toggleGroups[LAYERS.spaces], manifest.layers.normalized_spaces.available, true);
    this.setToggleState(this.toggleGroups[LAYERS.openings], manifest.layers.openings.available, true);
    this.setToggleState(this.toggleGroups[LAYERS.failed], manifest.layers.failed_entities.available, true);
    this.setToggleState(this.toggleGroups[LAYERS.shell], manifest.layers.envelope_shell.available, false);
    this.setToggleState(this.toggleGroups[LAYERS.surfaces], manifest.layers.surface_classification.available, true);
  }

  setToggleState(toggles, available, checkedByDefault) {
    for (const toggle of toggles) {
      toggle.disabled = !available;
      toggle.checked = available && checkedByDefault;
    }
  }

  isToggleEnabled(toggles) {
    return toggles.some((toggle) => !toggle.disabled && toggle.checked);
  }

  applyLayerVisibility() {
    for (const [layerId, root] of this.layerRoots.entries()) root.visible = this.isToggleEnabled(this.toggleGroups[layerId]);
    this.updateSelectionOverlay();
    this.renderEmptyState(this.hasVisibleContentLayer() ? EMPTY.ready : EMPTY.disabled);
  }

  hasVisibleContentLayer() {
    return [...this.layerRoots.entries()].some(([layerId, root]) => layerId !== LAYERS.grid && root?.visible);
  }

  handleCanvasClick(event) {
    if (!this.manifest || !this.hasVisibleContentLayer()) return;
    const rect = this.canvas.getBoundingClientRect();
    this.pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    this.pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
    this.raycaster.setFromCamera(this.pointer, this.camera);
    const keys = [];
    for (const hit of this.raycaster.intersectObjects(this.selectableMeshes, false)) {
      if (!isObjectVisible(hit.object)) continue;
      const key = hit.object.userData.selectKey;
      if (key && !keys.includes(key)) keys.push(key);
    }
    if (keys.length === 0) {
      this.lastClickSelection = null;
      return;
    }
    const clickPoint = { x: event.clientX, y: event.clientY };
    const index = this.isSameClickSelectionContext(clickPoint, keys) ? (this.lastClickSelection.index + 1) % keys.length : 0;
    this.lastClickSelection = { point: clickPoint, keys, index };
    this.selectEntity(keys[index], { scrollIntoView: true, revealSelection: true });
  }

  isSameClickSelectionContext(point, keys) {
    if (!this.lastClickSelection) return false;
    if (Math.abs(point.x - this.lastClickSelection.point.x) > 6 || Math.abs(point.y - this.lastClickSelection.point.y) > 6) return false;
    if (keys.length !== this.lastClickSelection.keys.length) return false;
    return keys.every((key, index) => key === this.lastClickSelection.keys[index]);
  }

  refreshEntityBrowser(scrollIntoView = false) {
    if (!this.browserList) return;
    this.browserList.innerHTML = "";
    this.listButtons = new Map();
    this.listRows = new Map();
    if (!this.manifest) {
      this.reportBrowserStats(emptyBrowserStats());
      this.renderBrowserEmptyState("No objects loaded yet.");
      return;
    }

    const entities = [...this.manifest.entities, ...(this.manifest.surface_entities || [])];
    this.reportBrowserStats(buildBrowserStats(entities, this.isPendingRemoval));
    const filtered = entities
      .filter((entity) => matchesBrowserFilter(entity, this.getBrowserFilter(), this.isPendingRemoval))
      .filter((entity) => matchesBrowserQuery(entity, this.getBrowserQuery()))
      .sort(compareBrowserEntities);

    if (filtered.length === 0) {
      this.renderBrowserEmptyState("No objects match the current search or filter.");
      return;
    }

    for (const entity of filtered) {
      const row = document.createElement("div");
      row.className = "object-row";
      row.dataset.state = entity.object_name === this.selectedKey
        ? "expanded"
        : this.highlightedKeys.includes(entity.object_name)
          ? "highlighted"
          : "idle";
      const button = document.createElement("button");
      button.type = "button";
      button.className = "object-button object-button--browser";
      button.dataset.key = entity.object_name;
      button.innerHTML = `
        <span class="object-button__header">
          <span class="object-button__title">${escapeHtml(entity.name || entity.object_name)}</span>
          <span class="object-badge" data-kind="${escapeHtml(browserBadgeKind(entity))}">${escapeHtml(browserBadgeLabel(entity))}</span>
        </span>
          <span class="object-button__meta">${escapeHtml(browserSecondaryText(entity))}</span>
          <span class="object-button__detail">${escapeHtml(describeEntityRow(entity))}</span>
        `;
      button.addEventListener("click", () => this.selectEntity(entity.object_name, { scrollIntoView: true }));
      row.appendChild(button);
      const actionButton = this.buildPendingRemovalToggle(entity);
      if (actionButton) row.appendChild(actionButton);
      if (entity.object_name === this.selectedKey) row.appendChild(this.buildInlineDetailsPanel(entity));
      this.browserList.appendChild(row);
      this.listButtons.set(entity.object_name, button);
      this.listRows.set(entity.object_name, row);
    }
    this.updateListSelection(scrollIntoView);
  }

  renderBrowserEmptyState(message) {
    const empty = document.createElement("p");
    empty.className = "viewer-selection-empty";
    empty.textContent = message;
    this.browserList.appendChild(empty);
  }

  reportBrowserStats(stats) {
    if (this.onBrowserStats) this.onBrowserStats(stats);
  }

  selectEntity(key, { scrollIntoView = false, revealSelection = false } = {}) {
    this.selectedKey = key;
    const entity = this.entityMap.get(key) || null;
    if (entity && revealSelection && this.revealSelectionInBrowser) {
      this.revealSelectionInBrowser(entity);
    }
    this.refreshEntityBrowser(scrollIntoView);
    this.renderInspectSummary(entity);
    this.updateSelectionOverlay();
    if (entity && this.onSelectionChange) this.onSelectionChange(entity);
  }

  updateListSelection(scrollIntoView) {
    for (const [key, button] of this.listButtons.entries()) {
      button.dataset.state = key === this.selectedKey ? "selected" : "idle";
      const row = this.listRows.get(key);
      if (row) {
        row.dataset.state = key === this.selectedKey ? "expanded" : this.highlightedKeys.includes(key) ? "highlighted" : "idle";
      }
      if (scrollIntoView && key === this.selectedKey) {
        (row || button).scrollIntoView({ block: "nearest" });
      }
    }
  }

  renderInspectSummary(entity) {
    if (this.inspectSelectionName) {
      this.inspectSelectionName.textContent = entity ? (entity.name || entity.object_name) : "No selection";
    }
    if (this.inspectSelectionMeta) {
      this.inspectSelectionMeta.textContent = entity ? browserSecondaryText(entity) : "Pick an object in the scene or object browser.";
    }
    if (!this.inspectSelectionActions) return;
    this.inspectSelectionActions.innerHTML = "";
    if (!isRemovalCandidate(entity) || !this.onTogglePendingRemoval) {
      this.inspectSelectionActions.classList.add("hidden");
      return;
    }
    this.inspectSelectionActions.appendChild(this.buildPrimaryRemovalAction(entity, "button button--ghost button--slate"));
    this.inspectSelectionActions.classList.remove("hidden");
  }

  buildInlineDetailsPanel(entity) {
    const panel = document.createElement("div");
    panel.className = "object-inline-panel";

    const detailsSection = document.createElement("section");
    detailsSection.className = "object-inline-panel__section";
    detailsSection.innerHTML = `<p class="subsection-label">Details</p>`;
    const detailsList = document.createElement("dl");
    detailsList.className = "stacked-detail-list object-inline-details";
    for (const [label, value] of detailFieldsForEntity(entity)) {
      const item = document.createElement("div");
      item.innerHTML = `<dt>${escapeHtml(label)}</dt><dd>${escapeHtml(value)}</dd>`;
      detailsList.appendChild(item);
    }
    detailsSection.appendChild(detailsList);
    panel.appendChild(detailsSection);

    const artifactsSection = document.createElement("section");
    artifactsSection.className = "object-inline-panel__section";
    artifactsSection.innerHTML = `<p class="subsection-label">Artifacts</p>`;
    const artifactsList = document.createElement("ul");
    artifactsList.className = "artifact-list artifact-list--stacked compact-artifacts object-inline-artifacts";
    const artifactPairs = artifactPairsForEntity(entity);
    if (artifactPairs.length === 0) {
      const item = document.createElement("li");
      item.textContent = "No artifact links for this object.";
      artifactsList.appendChild(item);
    } else {
      for (const [label, path] of artifactPairs) {
        const item = document.createElement("li");
        item.innerHTML = `<a href="/jobs/${this.currentJobId}/artifacts/${path}" target="_blank" rel="noreferrer">${escapeHtml(label)}</a><span>${escapeHtml(path)}</span>`;
        artifactsList.appendChild(item);
      }
    }
    artifactsSection.appendChild(artifactsList);
    panel.appendChild(artifactsSection);

    const actionsSection = document.createElement("section");
    actionsSection.className = "object-inline-panel__section";
    actionsSection.innerHTML = `<p class="subsection-label">Actions</p>`;
    const actionsWrap = document.createElement("div");
    actionsWrap.className = "selection-actions selection-actions--inline";
    const primaryAction = this.buildPrimaryRemovalAction(entity, "button button--ghost button--slate");
    if (primaryAction) {
      actionsWrap.appendChild(primaryAction);
    } else {
      const empty = document.createElement("p");
      empty.className = "viewer-selection-empty";
      empty.textContent = "No actions available for this object.";
      actionsWrap.appendChild(empty);
    }
    actionsSection.appendChild(actionsWrap);
    panel.appendChild(actionsSection);

    return panel;
  }

  buildPrimaryRemovalAction(entity, className) {
    if (!isRemovalCandidate(entity) || !this.onTogglePendingRemoval) return null;
    const actionButton = document.createElement("button");
    actionButton.type = "button";
    actionButton.className = className;
    actionButton.textContent = this.isMarkedForRemoval(entity) ? "Unmark Space Removal" : "Mark Space Removal";
    actionButton.disabled = !this.isRemovalToggleEnabled(entity);
    actionButton.addEventListener("click", () => this.onTogglePendingRemoval(entity));
    return actionButton;
  }

  buildPendingRemovalToggle(entity) {
    if (!isRemovalCandidate(entity) || !this.onTogglePendingRemoval) return null;
    const button = document.createElement("button");
    button.type = "button";
    button.className = "object-row__action";
    button.dataset.state = this.isMarkedForRemoval(entity) ? "marked" : "idle";
    button.textContent = this.isMarkedForRemoval(entity) ? "Unmark" : "Mark";
    button.disabled = !this.isRemovalToggleEnabled(entity);
    button.addEventListener("click", (event) => {
      event.stopPropagation();
      this.onTogglePendingRemoval(entity);
    });
    return button;
  }

  refreshRemovalControls() {
    if (!this.manifest) {
      this.reportBrowserStats(emptyBrowserStats());
      this.renderInspectSummary(null);
      return;
    }
    this.refreshEntityBrowser();
    this.renderInspectSummary(this.selectedKey ? this.entityMap.get(this.selectedKey) || null : null);
    this.updateSelectionOverlay();
  }

  setHighlightedKeys(keys) {
    const nextKeys = [...new Set((keys || []).filter(Boolean))];
    const unchanged = nextKeys.length === this.highlightedKeys.length && nextKeys.every((key, index) => key === this.highlightedKeys[index]);
    if (unchanged) return;
    this.highlightedKeys = nextKeys;
    this.refreshEntityBrowser();
    this.updateSelectionOverlay();
  }

  isMarkedForRemoval(entity) {
    return Boolean(this.isPendingRemoval && this.isPendingRemoval(entity));
  }

  isRemovalToggleEnabled(entity) {
    return Boolean(this.canTogglePendingRemoval ? this.canTogglePendingRemoval(entity) : true);
  }

  updateSelectionOverlay() {
    this.clearSelectionOverlay();
    for (const key of this.highlightedKeys) {
      if (!key || key === this.selectedKey) continue;
      const highlightedObject = this.findVisibleObject(LAYERS.failed, key) || this.findVisibleObject(LAYERS.surfaces, key) || this.findVisibleObject(LAYERS.openings, key) || this.findVisibleObject(LAYERS.spaces, key) || this.findVisibleObject(LAYERS.raw, key);
      if (!highlightedObject) continue;
      this.appendObjectOverlay(highlightedObject, this.entityMap.get(key) || null, {
        color: 0x3e63dd,
        edgeOpacity: 0.82,
        fillOpacity: 0.18,
        renderOrder: 950,
        scaleFactor: 1.45,
      });
    }
    if (!this.selectedKey) {
      return;
    }
    const selectedObject = this.findVisibleObject(LAYERS.failed, this.selectedKey) || this.findVisibleObject(LAYERS.surfaces, this.selectedKey) || this.findVisibleObject(LAYERS.openings, this.selectedKey) || this.findVisibleObject(LAYERS.spaces, this.selectedKey) || this.findVisibleObject(LAYERS.raw, this.selectedKey);
    if (!selectedObject) {
      return;
    }
    this.appendObjectOverlay(selectedObject, this.entityMap.get(this.selectedKey) || null, {
      color: 0xff7b3d,
      edgeOpacity: 0.95,
      fillOpacity: 0.22,
      renderOrder: 1000,
      scaleFactor: 1.75,
    });
  }

  findVisibleObject(layerId, key) {
    const object = this.layerObjectMaps[layerId].get(key);
    return object && isObjectVisible(object) ? object : null;
  }

  fitCamera() {
    const bounds = this.computeFocusBounds({ visibleOnly: true }) || this.focusBounds || this.contentBounds;
    if (!bounds) {
      this.resetCamera();
      return;
    }
    this.applyCameraFrame(buildCameraFrame(bounds));
  }

  resetCamera() {
    if (this.homeCameraFrame) {
      this.applyCameraFrame(this.homeCameraFrame);
      return;
    }
    this.applyCameraFrame({
      target: new Vector3(0, 0, 0),
      position: new Vector3(12, -10, 12),
      near: 0.01,
      far: 100000,
    });
  }

  resize() {
    const width = Math.max(this.canvas.parentElement.clientWidth, 320);
    const height = Math.max(this.canvas.parentElement.clientHeight, 320);
    this.renderer.setSize(width, height, false);
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
  }

  renderEmptyState(message) {
    this.emptyStateMessage.textContent = message;
    this.emptyState.classList.toggle("hidden", message === "");
  }

  disposeLayer(layerId) {
    const root = this.layerRoots.get(layerId);
    if (root) {
      root.removeFromParent();
      root.traverse((node) => {
        node.geometry?.dispose?.();
        if (Array.isArray(node.material)) {
          for (const material of node.material) material.dispose?.();
        } else {
          node.material?.dispose?.();
        }
      });
    }
    this.layerRoots.delete(layerId);
    this.focusRoots.delete(layerId);
    if (layerId === LAYERS.grid) {
      this.gridFloorRoot = null;
      this.gridFloorMaterial = null;
    }
    this.layerObjectMaps[layerId] = new Map();
  }

  buildGridFloor() {
    this.disposeLayer(LAYERS.grid);
    if (!(this.focusBounds || this.contentBounds)) return;
    const gridSpec = buildGridSpec(this.focusBounds || this.contentBounds);
    const root = new Group();
    const floor = new Mesh(new PlaneGeometry(1, 1), createWorldGridMaterial(gridSpec));
    floor.renderOrder = -10;
    root.add(floor);

    this.gridFloorRoot = root;
    this.gridFloorMaterial = floor.material;
    this.gridBaseSpan = gridSpec.baseSpan;
    this.layerRoots.set(LAYERS.grid, root);
    this.helperRoot.add(root);
    this.updateGridFloorPlacement();
  }

  computeContentBounds({ visibleOnly = false } = {}) {
    const bounds = new Box3();
    let hasContent = false;
    for (const [layerId, root] of this.layerRoots.entries()) {
      if (layerId === LAYERS.grid || !root) continue;
      if (visibleOnly && !isObjectVisible(root)) continue;
      bounds.expandByObject(root);
      hasContent = true;
    }
    return hasContent ? bounds : null;
  }

  computeFocusBounds({ visibleOnly = false } = {}) {
    const prioritizedRoots = [
      this.focusRoots.get(LAYERS.spaces) || null,
      this.focusRoots.get(LAYERS.raw) || null,
      this.focusRoots.get(LAYERS.shell) || null,
    ];
    for (const root of prioritizedRoots) {
      const bounds = this.boundsForRoot(root, visibleOnly);
      if (bounds) return bounds;
    }
    const fallbackRoots = [
      this.layerRoots.get(LAYERS.surfaces) || null,
      this.layerRoots.get(LAYERS.openings) || null,
      this.layerRoots.get(LAYERS.failed) || null,
    ];
    for (const root of fallbackRoots) {
      const bounds = this.boundsForRoot(root, visibleOnly);
      if (bounds) return bounds;
    }
    return null;
  }

  boundsForRoot(root, visibleOnly) {
    if (!root) return null;
    if (visibleOnly && !isObjectVisible(root)) return null;
    const bounds = new Box3().setFromObject(root);
    return bounds.isEmpty() ? null : bounds;
  }

  updateGridFloorPlacement() {
    if (!this.gridFloorRoot || !this.gridFloorMaterial) return;
    const cameraDistance = this.camera.position.distanceTo(this.controls.target);
    const span = chooseNiceStep(Math.max(cameraDistance * 5.5, this.gridBaseSpan, 120));
    this.gridFloorRoot.position.set(this.camera.position.x, this.camera.position.y, 0);
    this.gridFloorRoot.scale.set(span, span, 1);
    this.gridFloorMaterial.uniforms.uFadeStart.value = span * 0.16;
    this.gridFloorMaterial.uniforms.uFadeEnd.value = span * 0.52;
  }

  applyCameraFrame(frame) {
    this.controls.target.copy(frame.target);
    this.camera.position.copy(frame.position);
    this.camera.near = frame.near;
    this.camera.far = frame.far;
    this.camera.updateProjectionMatrix();
    this.controls.update();
  }

  appendObjectOverlay(selectedObject, entity, options) {
    if (entity && isMarkerOnlyFailedEntity(entity)) {
      const markerHighlight = this.buildMarkerSelectionOverlay(selectedObject, options);
      if (markerHighlight) this.selectionOverlayRoot.add(markerHighlight);
      return;
    }
    const overlays = [];
    selectedObject.updateWorldMatrix(true, true);
    selectedObject.traverse((node) => {
      if (!node.isMesh || !node.geometry) return;
      const edges = new LineSegments(new EdgesGeometry(node.geometry, 20), new LineBasicMaterial({ color: options.color, transparent: true, opacity: options.edgeOpacity, depthTest: false, depthWrite: false, toneMapped: false }));
      edges.matrixAutoUpdate = false;
      edges.matrix.copy(node.matrixWorld);
      edges.renderOrder = options.renderOrder;
      overlays.push(edges);
    });
    if (overlays.length === 0) {
      const markerHighlight = this.buildMarkerSelectionOverlay(selectedObject, options);
      if (markerHighlight) this.selectionOverlayRoot.add(markerHighlight);
      return;
    }
    for (const overlay of overlays) this.selectionOverlayRoot.add(overlay);
  }

  buildMarkerSelectionOverlay(selectedObject, options) {
    const sourceMesh = selectedObject?.isMesh ? selectedObject : selectedObject?.children.find((child) => child.isMesh) || null;
    if (!sourceMesh?.geometry) return null;
    sourceMesh.updateWorldMatrix(true, false);
    const highlight = new Mesh(sourceMesh.geometry, new MeshBasicMaterial({ color: options.color, transparent: true, opacity: options.fillOpacity, depthTest: false, depthWrite: false, toneMapped: false }));
    sourceMesh.matrixWorld.decompose(highlight.position, highlight.quaternion, highlight.scale);
    highlight.scale.multiplyScalar(options.scaleFactor);
    highlight.updateMatrix();
    highlight.matrixAutoUpdate = false;
    highlight.renderOrder = options.renderOrder - 1;
    return highlight;
  }

  clearSelectionOverlay() {
    for (const child of [...this.selectionOverlayRoot.children]) {
      child.removeFromParent();
      child.traverse((node) => {
        node.geometry?.dispose?.();
        if (Array.isArray(node.material)) {
          for (const material of node.material) material.dispose?.();
        } else {
          node.material?.dispose?.();
        }
      });
    }
  }
}

function normalizeElements(value) {
  if (!value) return [];
  if (Array.isArray(value)) return value.filter(Boolean);
  if (typeof value.length === "number" && !value.tagName) return Array.from(value).filter(Boolean);
  return [value];
}

function buildCameraFrame(bounds) {
  const center = bounds.getCenter(new Vector3());
  const size = bounds.getSize(new Vector3());
  const radius = Math.max(size.length() * 0.55, Math.max(size.x, size.y, size.z, 1) * 1.15, 1.8);
  const direction = new Vector3(1.15, -1.1, 0.82).normalize();
  return {
    target: center,
    position: center.clone().addScaledVector(direction, radius * 1.85),
    near: Math.max(radius / 1000, 0.01),
    far: Math.max(radius * 35, 100),
    radius,
  };
}

function buildGridSpec(bounds) {
  const size = bounds.getSize(new Vector3());
  const footprint = Math.max(size.x, size.y, 1);
  const paddedExtent = footprint + Math.max(footprint * 0.4, 6);
  const majorStep = chooseNiceStep(paddedExtent / 10);
  return {
    majorStep,
    minorStep: majorStep / 5,
    baseSpan: Math.max(majorStep * 24, footprint * 3.2, 180),
  };
}

function chooseNiceStep(value) {
  const magnitude = 10 ** Math.floor(Math.log10(Math.max(value, 0.1)));
  const normalized = value / magnitude;
  if (normalized <= 1) return magnitude;
  if (normalized <= 2) return 2 * magnitude;
  if (normalized <= 5) return 5 * magnitude;
  return 10 * magnitude;
}

function createWorldGridMaterial(gridSpec) {
  const material = new ShaderMaterial({
    transparent: true,
    depthWrite: false,
    side: DoubleSide,
    toneMapped: false,
    polygonOffset: true,
    polygonOffsetFactor: 1,
    polygonOffsetUnits: 1,
    uniforms: {
      uFloorColor: { value: new Color(0xf4f7fb) },
      uMinorColor: { value: new Color(0xd8e0e8) },
      uMajorColor: { value: new Color(0xa9b7c6) },
      uFloorOpacity: { value: 0.24 },
      uMinorOpacity: { value: 0.34 },
      uMajorOpacity: { value: 0.78 },
      uMinorStep: { value: gridSpec.minorStep },
      uMajorStep: { value: gridSpec.majorStep },
      uFadeStart: { value: gridSpec.baseSpan * 0.16 },
      uFadeEnd: { value: gridSpec.baseSpan * 0.52 },
    },
    vertexShader: `
      varying vec3 vWorldPosition;

      void main() {
        vec4 worldPosition = modelMatrix * vec4(position, 1.0);
        vWorldPosition = worldPosition.xyz;
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
      }
    `,
    fragmentShader: `
      uniform vec3 uFloorColor;
      uniform vec3 uMinorColor;
      uniform vec3 uMajorColor;
      uniform float uFloorOpacity;
      uniform float uMinorOpacity;
      uniform float uMajorOpacity;
      uniform float uMinorStep;
      uniform float uMajorStep;
      uniform float uFadeStart;
      uniform float uFadeEnd;

      varying vec3 vWorldPosition;

      float lineAxis(float coordinate, float stepSize) {
        float normalized = coordinate / stepSize;
        float centered = abs(fract(normalized - 0.5) - 0.5);
        float width = fwidth(normalized);
        return 1.0 - clamp(centered / max(width, 0.0001), 0.0, 1.0);
      }

      float gridLine(vec2 worldXY, float stepSize) {
        return max(lineAxis(worldXY.x, stepSize), lineAxis(worldXY.y, stepSize));
      }

      void main() {
        vec2 worldXY = vWorldPosition.xy;
        float radialDistance = length(worldXY - cameraPosition.xy);
        float fade = 1.0 - smoothstep(uFadeStart, uFadeEnd, radialDistance);
        if (fade <= 0.001) discard;

        float minorLine = gridLine(worldXY, uMinorStep) * uMinorOpacity;
        float majorLine = gridLine(worldXY, uMajorStep) * uMajorOpacity;
        vec3 color = mix(uFloorColor, uMinorColor, clamp(minorLine, 0.0, 1.0));
        color = mix(color, uMajorColor, clamp(majorLine, 0.0, 1.0));
        float alpha = max(uFloorOpacity, max(minorLine, majorLine)) * fade;
        gl_FragColor = vec4(color, alpha);
      }
    `,
  });
  material.extensions.derivatives = true;
  return material;
}

function isObjectVisible(object) {
  let current = object;
  while (current) {
    if (current.visible === false) return false;
    current = current.parent;
  }
  return true;
}

function isRemovalCandidate(entity) {
  return Boolean(entity && entity.selection_type === "entity" && entity.entity_type === "IfcSpace");
}

function browserGroup(entity) {
  if (entity.selection_type === "surface") return "surfaces";
  if (entity.failed) return "failed";
  if (entity.entity_type === "IfcSpace") return "spaces";
  if (entity.entity_type === "IfcOpeningElement") return "openings";
  return "other";
}

function compareBrowserEntities(left, right) {
  const order = { failed: 0, spaces: 1, openings: 2, surfaces: 3, other: 4 };
  const delta = order[browserGroup(left)] - order[browserGroup(right)];
  return delta || (left.name || left.object_name).localeCompare(right.name || right.object_name);
}

function browserBadgeKind(entity) {
  if (entity.failed) return "danger";
  if (entity.selection_type === "surface") return entity.classification || "surface";
  if (entity.entity_type === "IfcSpace") return "accent";
  if (entity.entity_type === "IfcOpeningElement") return "info";
  return "neutral";
}

function browserBadgeLabel(entity) {
  if (entity.selection_type === "surface") return entity.classification || "surface";
  if (entity.failed) return "failed";
  if (entity.entity_type === "IfcSpace") return entity.storey?.name || "space";
  if (entity.entity_type === "IfcOpeningElement") return "opening";
  return entity.entity_type || "entity";
}

function browserSecondaryText(entity) {
  const identifier = entity.selection_type === "surface" ? entity.surface_id || entity.object_name : entity.global_id || `#${entity.express_id}`;
  const location = entity.selection_type === "surface" ? entity.space_global_id || "-" : entity.storey?.name || entity.building?.name || entity.entity_type;
  return `${identifier} | ${location}`;
}

function describeEntityRow(entity) {
  if (entity.selection_type === "surface") {
    const area = typeof entity.area_m2 === "number" ? `${formatArea(entity.area_m2)} m2` : "-";
    return `${entity.classification || "surface"} | ${area}`;
  }
  if (entity.failed) return entity.reason || "Flagged geometry";
  if (Array.isArray(entity.clash_groups) && entity.clash_groups.length > 0) {
    const suffix = entity.recommended_clash_action ? ` | ${entity.recommended_clash_action}` : " | review required";
    return `${entity.clash_groups.length} clash group${entity.clash_groups.length === 1 ? "" : "s"}${suffix}`;
  }
  const location = entity.storey?.name || entity.building?.name;
  if (location) return location;
  if (typeof entity.volume_m3 === "number" && entity.volume_m3 > 0) return `${formatVolume(entity.volume_m3)} m3`;
  return entity.entity_type;
}

function buildBrowserStats(entities, isPendingRemoval) {
  return {
    all: entities.length,
    spaces: entities.filter((entity) => browserGroup(entity) === "spaces").length,
    openings: entities.filter((entity) => browserGroup(entity) === "openings").length,
    failed: entities.filter((entity) => browserGroup(entity) === "failed").length,
    surfaces: entities.filter((entity) => browserGroup(entity) === "surfaces").length,
    marked: entities.filter((entity) => Boolean(isPendingRemoval?.(entity))).length,
  };
}

function matchesBrowserFilter(entity, filterValue, isPendingRemoval) {
  const filter = (filterValue || "all").trim().toLowerCase();
  if (filter === "all") return true;
  if (filter === "marked") return Boolean(isPendingRemoval?.(entity));
  return browserGroup(entity) === filter;
}

function matchesBrowserQuery(entity, queryValue) {
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
    browserBadgeLabel(entity),
  ].filter(Boolean).join(" ").toLowerCase();
  return haystack.includes(query);
}

function isMarkerOnlyFailedEntity(entity) {
  return Boolean(entity?.failed && Array.isArray(entity.marker_origin) && !entity?.artifacts?.raw_obj && !entity?.artifacts?.normalized_obj);
}

function emptyBrowserStats() {
  return { all: 0, spaces: 0, openings: 0, failed: 0, surfaces: 0, marked: 0 };
}

function colorForEntity(objectName) {
  let hash = 0;
  for (const character of objectName) hash = (hash * 31 + character.charCodeAt(0)) >>> 0;
  const color = new Color();
  color.setHSL((hash % 360) / 360, 0.54, 0.54);
  return color;
}

function colorForSurfaceClassification(classification) {
  switch (classification) {
    case "external_wall":
      return 0x3e63dd;
    case "roof":
      return 0x0d9488;
    case "ground_floor":
      return 0x8a6d3b;
    case "internal_void":
      return 0xf59e0b;
    case "unclassified":
    default:
      return 0xda4757;
  }
}

function detailFieldsForEntity(entity) {
  return entity.selection_type === "surface"
    ? [
      ["Surface ID", entity.surface_id || entity.object_name],
      ["Space", entity.name || entity.object_name],
      ["Space GlobalId", entity.space_global_id || "-"],
      ["Classification", entity.classification || "-"],
      ["Area (m2)", formatArea(entity.area_m2)],
      ["Normal", formatOrigin(entity.normal)],
      ["Centroid", formatOrigin(entity.centroid)],
      ["Reason", entity.reason || "-"],
    ]
    : [
      ["Name", entity.name || entity.object_name],
      ["GlobalId", entity.global_id || "-"],
      ["Type", entity.entity_type],
      ["Storey", entity.storey?.name || "-"],
      ["Building", entity.building?.name || "-"],
      ["Valid", entity.valid ? "yes" : "no"],
      ["Failure Reason", entity.reason || "-"],
      ["Repair Status", entity.repair_status || "-"],
      ["Repair Backend", entity.repair_backend || "-"],
      ["Repair Reason", entity.repair_reason || "-"],
      ["Repair Actions", Array.isArray(entity.repair_actions) && entity.repair_actions.length ? entity.repair_actions.join(", ") : "-"],
      ["Clash Groups", Array.isArray(entity.clash_group_ids) && entity.clash_group_ids.length ? entity.clash_group_ids.join(", ") : "-"],
      ["Clash Types", Array.isArray(entity.clash_groups) && entity.clash_groups.length ? entity.clash_groups.map((group) => group.classification || "-").join(", ") : "-"],
      ["Recommended Clash Action", entity.recommended_clash_action || "-"],
      ["Faces", String(entity.face_count ?? 0)],
      ["Vertices", String(entity.vertex_count ?? 0)],
      ["Components", String(entity.component_count ?? 0)],
      ["Volume (m3)", formatVolume(entity.volume_m3)],
      ["Origin", formatOrigin(entity.marker_origin || entity.placement?.origin)],
    ];
}

function artifactPairsForEntity(entity) {
  return entity.selection_type === "surface"
    ? [
      ["All Surfaces OBJ", entity.artifacts?.classified_obj],
      ["Class OBJ", entity.artifacts?.class_obj],
      ["Shell OBJ", entity.artifacts?.shell_obj],
    ].filter(([, path]) => path)
    : [
      ["Raw OBJ", entity.artifacts?.raw_obj],
      ["Normalized OBJ", entity.artifacts?.normalized_obj],
      ["GLB", entity.artifacts?.glb],
    ].filter(([, path]) => path);
}

function formatOrigin(origin) {
  if (!Array.isArray(origin) || origin.length !== 3) return "-";
  return `(${origin.map((value) => Number.parseFloat(value).toFixed(3)).join(", ")})`;
}

function formatVolume(value) {
  if (typeof value !== "number" || Number.isNaN(value)) return "-";
  if (value === 0) return "0";
  return value < 0.001 ? value.toExponential(2) : value.toFixed(3);
}

function formatArea(value) {
  if (typeof value !== "number" || Number.isNaN(value)) return "-";
  if (value === 0) return "0";
  return value < 0.001 ? value.toExponential(2) : value.toFixed(3);
}
