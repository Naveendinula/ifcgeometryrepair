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
  MeshStandardMaterial,
  PerspectiveCamera,
  Raycaster,
  Scene,
  SphereGeometry,
  Vector2,
  Vector3,
  WebGLRenderer,
} from "three";
import { OrbitControls } from "/static/vendor/three/examples/jsm/controls/OrbitControls.js";
import { OBJLoader } from "/static/vendor/three/examples/jsm/loaders/OBJLoader.js";
import { GLTFLoader } from "/static/vendor/three/examples/jsm/loaders/GLTFLoader.js";


const LAYER_IDS = {
  raw: "raw",
  normalizedSpaces: "normalizedSpaces",
  openings: "openings",
  failed: "failed",
};

const EMPTY_MESSAGES = {
  initial: "Use the import control above to create a job and populate the 3D workspace.",
  loading: "Loading viewer artifacts...",
  unavailable: "No viewer geometry is available for this job.",
  disabled: "Enable at least one layer to inspect the job geometry.",
  ready: "",
};


export class DebugViewer {
  constructor(options) {
    this.panel = options.panel;
    this.manifestLink = options.manifestLink;
    this.canvas = options.canvas;
    this.emptyState = options.emptyState;
    this.emptyStateMessage = options.emptyStateMessage || options.emptyState;
    this.selectionEmpty = options.selectionEmpty;
    this.metadataList = options.metadataList;
    this.artifactLinks = options.artifactLinks;
    this.spaceList = options.spaceList;
    this.openingList = options.openingList;
    this.failedList = options.failedList;
    this.onError = options.onError;

    this.toggleGroups = {
      [LAYER_IDS.raw]: normalizeElements(options.toggleRaw),
      [LAYER_IDS.normalizedSpaces]: normalizeElements(options.toggleNormalized),
      [LAYER_IDS.openings]: normalizeElements(options.toggleOpenings),
      [LAYER_IDS.failed]: normalizeElements(options.toggleFailed),
    };
    this.resetViewButtons = normalizeElements(options.resetViewButtons);
    this.fitViewButtons = normalizeElements(options.fitViewButtons);

    this.currentJobId = null;
    this.manifest = null;
    this.selectedKey = null;
    this.entityMap = new Map();
    this.listButtons = new Map();
    this.selectableMeshes = [];
    this.layerRoots = new Map();
    this.layerObjectMaps = {
      [LAYER_IDS.raw]: new Map(),
      [LAYER_IDS.normalizedSpaces]: new Map(),
      [LAYER_IDS.openings]: new Map(),
      [LAYER_IDS.failed]: new Map(),
    };

    this.objLoader = new OBJLoader();
    this.gltfLoader = new GLTFLoader();
    this.raycaster = new Raycaster();
    this.pointer = new Vector2();

    this.scene = new Scene();
    this.camera = new PerspectiveCamera(52, 1, 0.01, 100000);
    this.camera.position.set(12, 10, 12);

    this.renderer = new WebGLRenderer({
      canvas: this.canvas,
      antialias: true,
      alpha: true,
    });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));

    this.controls = new OrbitControls(this.camera, this.canvas);
    this.controls.enableDamping = true;
    this.controls.target.set(0, 0, 0);

    this.scene.add(new AmbientLight(0xffffff, 1.35));
    const keyLight = new DirectionalLight(0xffffff, 1.15);
    keyLight.position.set(12, 20, 8);
    this.scene.add(keyLight);

    const fillLight = new DirectionalLight(0xe4edf5, 0.55);
    fillLight.position.set(-8, 10, -10);
    this.scene.add(fillLight);

    this.selectionOverlayRoot = new Group();
    this.selectionOverlayRoot.name = "selection-overlay";
    this.scene.add(this.selectionOverlayRoot);

    for (const toggles of Object.values(this.toggleGroups)) {
      this.bindToggleGroup(toggles);
    }
    for (const button of this.resetViewButtons) {
      button?.addEventListener("click", () => this.resetCamera());
    }
    for (const button of this.fitViewButtons) {
      button?.addEventListener("click", () => this.fitCamera());
    }

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
      this.clear(EMPTY_MESSAGES.unavailable);
      return;
    }

    if (this.currentJobId === jobId && this.manifest) {
      this.setSceneActionState(true);
      this.resize();
      return;
    }

    this.clear(EMPTY_MESSAGES.loading);

    const manifestUrl = `/jobs/${jobId}/artifacts/${manifestPath}`;
    const response = await fetch(manifestUrl);
    const manifest = await response.json();
    if (!response.ok) {
      this.clear(EMPTY_MESSAGES.unavailable);
      throw new Error(manifest.detail || "Could not load viewer manifest.");
    }

    this.currentJobId = jobId;
    this.manifest = manifest;
    this.manifestLink.href = manifestUrl;
    this.manifestLink.classList.remove("hidden");

    this.entityMap = new Map(manifest.entities.map((entity) => [entity.object_name, entity]));

    await this.loadLayers(jobId, manifest);
    this.buildFailedLayer();
    this.renderEntityLists();
    this.configureToggles(manifest);
    this.setSceneActionState(true);
    this.applyLayerVisibility();
    this.renderMetadata(null);
    this.resize();
    this.fitCamera();
  }

  clear(message = EMPTY_MESSAGES.initial) {
    this.currentJobId = null;
    this.manifest = null;
    this.selectedKey = null;
    this.entityMap = new Map();
    this.listButtons = new Map();
    this.selectableMeshes = [];

    this.manifestLink.href = "#";
    this.manifestLink.classList.add("hidden");

    this.renderList(this.spaceList, []);
    this.renderList(this.openingList, []);
    this.renderList(this.failedList, []);
    this.renderMetadata(null);

    for (const layerId of Object.values(LAYER_IDS)) {
      this.disposeLayer(layerId);
    }

    this.clearSelectionOverlay();
    this.setToggleState(this.toggleGroups[LAYER_IDS.raw], false, false);
    this.setToggleState(this.toggleGroups[LAYER_IDS.normalizedSpaces], false, false);
    this.setToggleState(this.toggleGroups[LAYER_IDS.openings], false, false);
    this.setToggleState(this.toggleGroups[LAYER_IDS.failed], false, false);
    this.setSceneActionState(false);
    this.renderEmptyState(message);
    this.resetCamera();
  }

  animate() {
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
    window.requestAnimationFrame(this.animate);
  }

  bindToggleGroup(toggles) {
    for (const toggle of toggles) {
      toggle.addEventListener("change", () => {
        this.syncToggleGroup(toggles, toggle.checked);
        this.applyLayerVisibility();
      });
    }
  }

  syncToggleGroup(toggles, checked) {
    for (const toggle of toggles) {
      if (toggle.disabled) {
        toggle.checked = false;
        continue;
      }
      toggle.checked = checked;
    }
  }

  setSceneActionState(enabled) {
    for (const button of [...this.resetViewButtons, ...this.fitViewButtons]) {
      if (button) {
        button.disabled = !enabled;
      }
    }
  }

  async loadLayers(jobId, manifest) {
    await Promise.all([
      this.loadRawLayer(jobId, manifest.layers.raw_ifc_preview),
      this.loadAggregateLayer(jobId, LAYER_IDS.normalizedSpaces, manifest.layers.normalized_spaces),
      this.loadAggregateLayer(jobId, LAYER_IDS.openings, manifest.layers.openings),
    ]);
  }

  async loadRawLayer(jobId, layerConfig) {
    if (!layerConfig?.available) {
      return;
    }

    const root = new Group();
    const assetPaths = [layerConfig.spaces_obj, layerConfig.openings_obj].filter(Boolean);
    for (const assetPath of assetPaths) {
      const object = await this.loadAsset(jobId, assetPath, null);
      root.add(object);
      this.indexNamedRoots(object, LAYER_IDS.raw);
    }

    this.layerRoots.set(LAYER_IDS.raw, root);
    this.scene.add(root);
  }

  async loadAggregateLayer(jobId, layerId, layerConfig) {
    if (!layerConfig?.available) {
      return;
    }

    const object = await this.loadAsset(jobId, layerConfig.obj, layerConfig.glb);
    this.layerRoots.set(layerId, object);
    this.scene.add(object);
    this.indexNamedRoots(object, layerId);
  }

  async loadAsset(jobId, objPath, glbPath) {
    const artifactPath = glbPath || objPath;
    if (!artifactPath) {
      return new Group();
    }

    const artifactUrl = `/jobs/${jobId}/artifacts/${artifactPath}`;
    if (glbPath) {
      const gltf = await this.loadGlb(artifactUrl);
      return gltf.scene;
    }

    return this.loadObj(artifactUrl);
  }

  loadObj(url) {
    return new Promise((resolve, reject) => {
      this.objLoader.load(url, resolve, undefined, reject);
    });
  }

  loadGlb(url) {
    return new Promise((resolve, reject) => {
      this.gltfLoader.load(url, resolve, undefined, reject);
    });
  }

  indexNamedRoots(root, layerId) {
    for (const namedRoot of this.collectNamedRoots(root)) {
      const key = namedRoot.name;
      const entity = this.entityMap.get(key);
      if (!entity) {
        continue;
      }

      namedRoot.userData.selectKey = key;
      namedRoot.userData.layerId = layerId;
      this.layerObjectMaps[layerId].set(key, namedRoot);
      this.applyStyle(namedRoot, layerId, entity);

      namedRoot.traverse((node) => {
        if (!node.isMesh) {
          return;
        }
        node.userData.selectKey = key;
        node.userData.layerId = layerId;
        this.selectableMeshes.push(node);
      });
    }
  }

  collectNamedRoots(root) {
    const namedRoots = [];

    const visit = (node, isRoot) => {
      if (!isRoot && this.entityMap.has(node.name)) {
        namedRoots.push(node);
        return;
      }

      for (const child of node.children) {
        visit(child, false);
      }
    };

    visit(root, true);
    return namedRoots;
  }

  applyStyle(root, layerId, entity) {
    root.traverse((node) => {
      if (!node.isMesh) {
        return;
      }
      node.material = this.createMaterial(layerId, entity);
    });
  }

  createMaterial(layerId, entity) {
    if (layerId === LAYER_IDS.raw) {
      return new MeshBasicMaterial({
        color: 0x95a3b2,
        transparent: true,
        opacity: 0.2,
        depthWrite: false,
        side: DoubleSide,
      });
    }

    if (layerId === LAYER_IDS.failed) {
      return new MeshBasicMaterial({
        color: 0xda4757,
        transparent: true,
        opacity: 0.58,
        depthWrite: false,
        wireframe: true,
        side: DoubleSide,
      });
    }

    if (layerId === LAYER_IDS.openings) {
      return new MeshStandardMaterial({
        color: 0x3259d7,
        transparent: true,
        opacity: 0.9,
        roughness: 0.58,
        metalness: 0.0,
        side: DoubleSide,
      });
    }

    return new MeshStandardMaterial({
      color: this.colorForEntity(entity.object_name),
      transparent: true,
      opacity: 0.92,
      roughness: 0.72,
      metalness: 0.02,
      side: DoubleSide,
    });
  }

  buildFailedLayer() {
    this.disposeLayer(LAYER_IDS.failed);

    if (!this.manifest) {
      return;
    }

    const failedRoot = new Group();
    const failedEntities = this.manifest.entities.filter((entity) => entity.failed);

    for (const entity of failedEntities) {
      const key = entity.object_name;
      const source =
        this.layerObjectMaps[LAYER_IDS.normalizedSpaces].get(key) ||
        this.layerObjectMaps[LAYER_IDS.openings].get(key) ||
        this.layerObjectMaps[LAYER_IDS.raw].get(key);

      if (source) {
        const clone = source.clone(true);
        clone.name = key;
        clone.userData.selectKey = key;
        clone.userData.layerId = LAYER_IDS.failed;
        this.applyStyle(clone, LAYER_IDS.failed, entity);
        this.layerObjectMaps[LAYER_IDS.failed].set(key, clone);
        clone.traverse((node) => {
          if (!node.isMesh) {
            return;
          }
          node.userData.selectKey = key;
          node.userData.layerId = LAYER_IDS.failed;
          this.selectableMeshes.push(node);
        });
        failedRoot.add(clone);
        continue;
      }

      if (Array.isArray(entity.marker_origin)) {
        const marker = new Mesh(
          new SphereGeometry(0.15, 16, 16),
          new MeshBasicMaterial({
            color: 0xda4757,
            transparent: true,
            opacity: 0.95,
          }),
        );
        marker.position.set(entity.marker_origin[0], entity.marker_origin[1], entity.marker_origin[2]);
        marker.name = key;
        marker.userData.selectKey = key;
        marker.userData.layerId = LAYER_IDS.failed;
        this.layerObjectMaps[LAYER_IDS.failed].set(key, marker);
        this.selectableMeshes.push(marker);
        failedRoot.add(marker);
      }
    }

    this.layerRoots.set(LAYER_IDS.failed, failedRoot);
    this.scene.add(failedRoot);
  }

  configureToggles(manifest) {
    this.setToggleState(this.toggleGroups[LAYER_IDS.raw], manifest.layers.raw_ifc_preview.available, false);
    this.setToggleState(
      this.toggleGroups[LAYER_IDS.normalizedSpaces],
      manifest.layers.normalized_spaces.available,
      true,
    );
    this.setToggleState(this.toggleGroups[LAYER_IDS.openings], manifest.layers.openings.available, true);
    this.setToggleState(this.toggleGroups[LAYER_IDS.failed], manifest.layers.failed_entities.available, true);
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
    const rawRoot = this.layerRoots.get(LAYER_IDS.raw);
    const normalizedRoot = this.layerRoots.get(LAYER_IDS.normalizedSpaces);
    const openingRoot = this.layerRoots.get(LAYER_IDS.openings);
    const failedRoot = this.layerRoots.get(LAYER_IDS.failed);

    if (rawRoot) {
      rawRoot.visible = this.isToggleEnabled(this.toggleGroups[LAYER_IDS.raw]);
    }
    if (normalizedRoot) {
      normalizedRoot.visible = this.isToggleEnabled(this.toggleGroups[LAYER_IDS.normalizedSpaces]);
    }
    if (openingRoot) {
      openingRoot.visible = this.isToggleEnabled(this.toggleGroups[LAYER_IDS.openings]);
    }
    if (failedRoot) {
      failedRoot.visible = this.isToggleEnabled(this.toggleGroups[LAYER_IDS.failed]);
    }

    this.updateSelectionOverlay();
    if (!this.hasVisibleLayer()) {
      this.renderEmptyState(EMPTY_MESSAGES.disabled);
      return;
    }

    this.renderEmptyState(EMPTY_MESSAGES.ready);
  }

  hasVisibleLayer() {
    return (
      (this.layerRoots.get(LAYER_IDS.raw)?.visible ?? false) ||
      (this.layerRoots.get(LAYER_IDS.normalizedSpaces)?.visible ?? false) ||
      (this.layerRoots.get(LAYER_IDS.openings)?.visible ?? false) ||
      (this.layerRoots.get(LAYER_IDS.failed)?.visible ?? false)
    );
  }

  handleCanvasClick(event) {
    if (!this.manifest || !this.hasVisibleLayer()) {
      return;
    }

    const rect = this.canvas.getBoundingClientRect();
    this.pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    this.pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
    this.raycaster.setFromCamera(this.pointer, this.camera);

    const intersections = this.raycaster.intersectObjects(this.selectableMeshes, false);
    const visibleHit = intersections.find((hit) => this.isObjectVisible(hit.object));
    if (!visibleHit) {
      return;
    }

    const selectKey = visibleHit.object.userData.selectKey;
    if (selectKey) {
      this.selectEntity(selectKey, true);
    }
  }

  isObjectVisible(object) {
    let current = object;
    while (current) {
      if (current.visible === false) {
        return false;
      }
      current = current.parent;
    }
    return true;
  }

  renderEntityLists() {
    if (!this.manifest) {
      return;
    }

    const spaces = this.manifest.entities
      .filter((entity) => entity.entity_type === "IfcSpace" && !entity.failed)
      .sort(compareEntityLabels);
    const openings = this.manifest.entities
      .filter((entity) => entity.entity_type === "IfcOpeningElement" && !entity.failed)
      .sort(compareEntityLabels);
    const failedEntities = this.manifest.entities.filter((entity) => entity.failed).sort(compareEntityLabels);

    this.listButtons = new Map();
    this.renderList(this.spaceList, spaces);
    this.renderList(this.openingList, openings);
    this.renderList(this.failedList, failedEntities);
  }

  renderList(container, entities) {
    container.innerHTML = "";
    if (entities.length === 0) {
      const empty = document.createElement("p");
      empty.className = "viewer-selection-empty";
      empty.textContent = "None";
      container.appendChild(empty);
      return;
    }

    for (const entity of entities) {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "object-button";
      button.dataset.key = entity.object_name;
      button.innerHTML = `
        <span class="object-button__title">${entity.name || entity.object_name}</span>
        <small>${entity.global_id || `#${entity.express_id}`}</small>
        <small>${describeEntityRow(entity)}</small>
      `;
      button.addEventListener("click", () => this.selectEntity(entity.object_name, true));
      container.appendChild(button);
      this.listButtons.set(entity.object_name, button);
    }
  }

  selectEntity(key, scrollIntoView = false) {
    this.selectedKey = key;
    this.updateListSelection(scrollIntoView);
    this.renderMetadata(this.entityMap.get(key) || null);
    this.updateSelectionOverlay();
  }

  updateListSelection(scrollIntoView) {
    for (const [key, button] of this.listButtons.entries()) {
      button.dataset.state = key === this.selectedKey ? "selected" : "idle";
      if (scrollIntoView && key === this.selectedKey) {
        button.scrollIntoView({ block: "nearest" });
      }
    }
  }

  renderMetadata(entity) {
    this.metadataList.innerHTML = "";
    this.artifactLinks.innerHTML = "";

    if (!entity) {
      this.selectionEmpty.classList.remove("hidden");
      return;
    }

    this.selectionEmpty.classList.add("hidden");
    const fields = [
      ["Name", entity.name || entity.object_name],
      ["GlobalId", entity.global_id || "-"],
      ["Type", entity.entity_type],
      ["Storey", entity.storey?.name || "-"],
      ["Building", entity.building?.name || "-"],
      ["Valid", entity.valid ? "yes" : "no"],
      ["Failure Reason", entity.reason || "-"],
      ["Faces", String(entity.face_count ?? 0)],
      ["Vertices", String(entity.vertex_count ?? 0)],
      ["Components", String(entity.component_count ?? 0)],
      ["Volume (m3)", formatVolume(entity.volume_m3)],
      ["Origin", formatOrigin(entity.marker_origin || entity.placement?.origin)],
    ];

    for (const [label, value] of fields) {
      const item = document.createElement("div");
      const dt = document.createElement("dt");
      const dd = document.createElement("dd");
      dt.textContent = label;
      dd.textContent = value;
      item.append(dt, dd);
      this.metadataList.appendChild(item);
    }

    const artifactPairs = [
      ["Raw OBJ", entity.artifacts?.raw_obj],
      ["Normalized OBJ", entity.artifacts?.normalized_obj],
      ["GLB", entity.artifacts?.glb],
    ].filter(([, path]) => path);

    if (artifactPairs.length === 0) {
      const item = document.createElement("li");
      item.textContent = "No artifact links for the selected object.";
      this.artifactLinks.appendChild(item);
      return;
    }

    for (const [label, path] of artifactPairs) {
      const item = document.createElement("li");
      const link = document.createElement("a");
      link.href = `/jobs/${this.currentJobId}/artifacts/${path}`;
      link.textContent = label;
      link.target = "_blank";
      link.rel = "noreferrer";
      const value = document.createElement("span");
      value.textContent = path;
      item.append(link, value);
      this.artifactLinks.appendChild(item);
    }
  }

  updateSelectionOverlay() {
    if (!this.selectedKey) {
      this.clearSelectionOverlay();
      return;
    }

    const selectedObject =
      this.findVisibleObject(LAYER_IDS.failed, this.selectedKey) ||
      this.findVisibleObject(LAYER_IDS.openings, this.selectedKey) ||
      this.findVisibleObject(LAYER_IDS.normalizedSpaces, this.selectedKey) ||
      this.findVisibleObject(LAYER_IDS.raw, this.selectedKey);

    if (!selectedObject) {
      this.clearSelectionOverlay();
      return;
    }

    this.rebuildSelectionOverlay(selectedObject, this.entityMap.get(this.selectedKey) || null);
  }

  findVisibleObject(layerId, key) {
    const object = this.layerObjectMaps[layerId].get(key);
    if (!object) {
      return null;
    }
    return this.isObjectVisible(object) ? object : null;
  }

  fitCamera() {
    const bounds = new Box3();
    let hasContent = false;

    for (const root of this.layerRoots.values()) {
      if (!root || !root.visible) {
        continue;
      }
      bounds.expandByObject(root);
      hasContent = true;
    }

    if (!hasContent) {
      this.resetCamera();
      return;
    }

    const center = bounds.getCenter(new Vector3());
    const size = bounds.getSize(new Vector3());
    const radius = Math.max(size.length() * 0.55, 1.2);

    this.controls.target.copy(center);
    this.camera.position.set(center.x + radius, center.y + radius * 0.8, center.z + radius);
    this.camera.near = Math.max(radius / 1000, 0.01);
    this.camera.far = Math.max(radius * 25, 100);
    this.camera.updateProjectionMatrix();
    this.controls.update();
  }

  resetCamera() {
    this.controls.target.set(0, 0, 0);
    this.camera.position.set(12, 10, 12);
    this.camera.near = 0.01;
    this.camera.far = 100000;
    this.camera.updateProjectionMatrix();
    this.controls.update();
  }

  resize() {
    const width = Math.max(this.canvas.parentElement.clientWidth, 320);
    const height = Math.max(this.canvas.parentElement.clientHeight, 320);
    this.renderer.setSize(width, height, false);
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
  }

  renderEmptyState(message) {
    if (this.emptyStateMessage !== this.emptyState) {
      this.emptyStateMessage.textContent = message;
    } else {
      this.emptyState.textContent = message;
    }
    this.emptyState.classList.toggle("hidden", message === "");
  }

  disposeLayer(layerId) {
    const root = this.layerRoots.get(layerId);
    if (root) {
      root.removeFromParent();
      root.traverse((node) => {
        if (node.geometry?.dispose) {
          node.geometry.dispose();
        }
        if (Array.isArray(node.material)) {
          for (const material of node.material) {
            material.dispose?.();
          }
        } else if (node.material?.dispose) {
          node.material.dispose();
        }
      });
    }

    this.layerRoots.delete(layerId);
    this.layerObjectMaps[layerId] = new Map();
  }

  colorForEntity(objectName) {
    let hash = 0;
    for (const character of objectName) {
      hash = (hash * 31 + character.charCodeAt(0)) >>> 0;
    }

    const hue = hash % 360;
    const color = new Color();
    color.setHSL(hue / 360, 0.54, 0.54);
    return color;
  }

  rebuildSelectionOverlay(selectedObject, entity) {
    this.clearSelectionOverlay();

    if (entity && isMarkerOnlyFailedEntity(entity)) {
      const markerHighlight = this.buildMarkerSelectionOverlay(selectedObject);
      if (markerHighlight) {
        this.selectionOverlayRoot.add(markerHighlight);
      }
      return;
    }

    const overlays = [];
    selectedObject.updateWorldMatrix(true, true);
    selectedObject.traverse((node) => {
      if (!node.isMesh || !node.geometry) {
        return;
      }

      const edgesGeometry = new EdgesGeometry(node.geometry, 20);
      const edgeLines = new LineSegments(
        edgesGeometry,
        new LineBasicMaterial({
          color: 0xff7b3d,
          transparent: true,
          opacity: 0.95,
          depthTest: false,
          depthWrite: false,
          toneMapped: false,
        }),
      );
      edgeLines.matrixAutoUpdate = false;
      edgeLines.matrix.copy(node.matrixWorld);
      edgeLines.renderOrder = 1000;
      overlays.push(edgeLines);
    });

    if (overlays.length === 0) {
      const markerHighlight = this.buildMarkerSelectionOverlay(selectedObject);
      if (markerHighlight) {
        this.selectionOverlayRoot.add(markerHighlight);
      }
      return;
    }

    for (const overlay of overlays) {
      this.selectionOverlayRoot.add(overlay);
    }
  }

  buildMarkerSelectionOverlay(selectedObject) {
    if (!selectedObject) {
      return null;
    }

    const sourceMesh = selectedObject.isMesh
      ? selectedObject
      : selectedObject.children.find((child) => child.isMesh) || null;
    if (!sourceMesh || !sourceMesh.geometry) {
      return null;
    }

    sourceMesh.updateWorldMatrix(true, false);
    const highlight = new Mesh(
      sourceMesh.geometry,
      new MeshBasicMaterial({
        color: 0xff7b3d,
        transparent: true,
        opacity: 0.22,
        depthTest: false,
        depthWrite: false,
        toneMapped: false,
      }),
    );
    sourceMesh.matrixWorld.decompose(highlight.position, highlight.quaternion, highlight.scale);
    highlight.scale.multiplyScalar(1.75);
    highlight.updateMatrix();
    highlight.matrixAutoUpdate = false;
    highlight.renderOrder = 999;
    return highlight;
  }

  clearSelectionOverlay() {
    for (const child of [...this.selectionOverlayRoot.children]) {
      child.removeFromParent();
      child.traverse((node) => {
        node.geometry?.dispose?.();
        if (Array.isArray(node.material)) {
          for (const material of node.material) {
            material.dispose?.();
          }
        } else {
          node.material?.dispose?.();
        }
      });
    }
  }
}


function normalizeElements(value) {
  if (!value) {
    return [];
  }
  if (Array.isArray(value)) {
    return value.filter(Boolean);
  }
  if (typeof value.length === "number" && !value.tagName) {
    return Array.from(value).filter(Boolean);
  }
  return [value];
}


function describeEntityRow(entity) {
  if (entity.failed) {
    return entity.reason || "Flagged geometry";
  }

  const location = entity.storey?.name || entity.building?.name;
  if (location) {
    return location;
  }

  if (typeof entity.volume_m3 === "number" && entity.volume_m3 > 0) {
    return `${formatVolume(entity.volume_m3)} m3`;
  }

  return entity.entity_type;
}


function isMarkerOnlyFailedEntity(entity) {
  return Boolean(
    entity?.failed &&
      Array.isArray(entity.marker_origin) &&
      !entity?.artifacts?.raw_obj &&
      !entity?.artifacts?.normalized_obj,
  );
}


function compareEntityLabels(left, right) {
  return (left.name || left.object_name).localeCompare(right.name || right.object_name);
}


function formatOrigin(origin) {
  if (!Array.isArray(origin) || origin.length !== 3) {
    return "-";
  }
  return `(${origin.map((value) => Number.parseFloat(value).toFixed(3)).join(", ")})`;
}


function formatVolume(value) {
  if (typeof value !== "number" || Number.isNaN(value)) {
    return "-";
  }
  if (value === 0) {
    return "0";
  }
  return value < 0.001 ? value.toExponential(2) : value.toFixed(3);
}
