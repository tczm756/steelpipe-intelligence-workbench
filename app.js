import * as THREE from "./vendor/three.module.min.js";
import * as ort from "./vendor/onnxruntime/ort.wasm.min.mjs";

const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => Array.from(root.querySelectorAll(selector));
const clamp = (value, min, max) => Math.min(max, Math.max(min, value));
const prefersReducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
const SVG_NS = "http://www.w3.org/2000/svg";

ort.env.wasm.wasmPaths = new URL("./vendor/onnxruntime/", window.location.href).href;
ort.env.wasm.numThreads = 1;
ort.env.wasm.proxy = true;
ort.env.logLevel = "error";

const compressedRuntimeReady = prepareCompressedRuntime();

navigator.serviceWorker?.addEventListener("message", (event) => {
  if (event.data?.type === "compressed-wasm-served") {
    window.__compressedWasmServed = true;
  }
});

const segmentationModels = {
  "small-unet-boundary": {
    id: "small-unet-boundary",
    name: "SmallUNet + 边界感知损失",
    badge: "BOUNDARY WEIGHT",
    manifest: null,
    session: null,
  },
  "small-unet-original": {
    id: "small-unet-original",
    name: "SmallUNet 原始最佳",
    badge: "ORIGINAL WEIGHT",
    manifest: null,
    session: null,
  },
  "small-unet-ablation": {
    id: "small-unet-ablation",
    name: "SmallUNet 消融最佳",
    badge: "ABLATION WEIGHT",
    manifest: null,
    session: null,
  },
};

const recognitionModels = {
  mobilenetv2: {
    id: "mobilenetv2",
    name: "MobileNetV2",
    badge: "BENCHMARK WEIGHT",
    manifest: null,
    session: null,
  },
  "small-classifier": {
    id: "small-classifier",
    name: "SmallDefectClassifier 消融最佳",
    badge: "ABLATION WEIGHT",
    manifest: null,
    session: null,
  },
  "small-classifier-original": {
    id: "small-classifier-original",
    name: "SmallDefectClassifier 原始最佳",
    badge: "ORIGINAL WEIGHT",
    manifest: null,
    session: null,
  },
};

const trainingStages = [
  { id: "data", label: "数据锁定", short: "01" },
  { id: "weights", label: "权重装载", short: "02" },
  { id: "loop", label: "训练循环", short: "03" },
  { id: "validation", label: "验证评估", short: "04" },
  { id: "export", label: "导出归档", short: "05" },
];

const archiveRecords = [
  {
    id: "benchmark_v1",
    name: "benchmark_v1",
    suite: "segmentation",
    model: "SmallUNet + 边界感知损失",
    status: "completed",
    date: "2026-11-14 12:30",
    primaryLabel: "Dice",
    primaryValue: 87.55,
    hardware: "RTX 4090D",
    duration: "2h 56m",
    runs: 33,
    weights: { seg: "small-unet-boundary", rec: "mobilenetv2" },
    note: "33 次正式训练，固定切分与三种子评测。",
  },
  {
    id: "ablation_2026-11-09",
    name: "ablation_2026-11-09",
    suite: "segmentation",
    model: "SmallUNet 消融",
    status: "completed",
    date: "2026-11-09 23:48",
    primaryLabel: "Dice",
    primaryValue: 87.61,
    hardware: "RTX 3080 Ti",
    duration: "1h 42m",
    runs: 12,
    weights: { seg: "small-unet-ablation", rec: "small-classifier" },
    note: "Plain U-Net / SmallUNet 与单双通道三随机种子对照。",
  },
  {
    id: "line-scan-live-047",
    name: "line-scan-live-047",
    suite: "recognition",
    model: "MobileNetV2",
    status: "running",
    date: "2026-11-15 14:02",
    primaryLabel: "Accuracy",
    primaryValue: 99.18,
    hardware: "Cloud A100",
    duration: "18m",
    runs: 8,
    weights: { seg: "small-unet-boundary", rec: "mobilenetv2" },
    note: "现场线激光数据适配，按钢管编号分组。",
  },
  {
    id: "segmentation-demo",
    name: "segmentation-demo",
    suite: "segmentation",
    model: "SmallUNet · seed 44",
    status: "completed",
    date: "2026-11-09 18:16",
    primaryLabel: "Dice",
    primaryValue: 87.47,
    hardware: "Local CUDA",
    duration: "4m 32s",
    runs: 1,
    weights: { seg: "small-unet-original", rec: "small-classifier-original" },
    note: "单图推理复现，输出概率图、掩码与叠加图。",
  },
  {
    id: "cbam-stability-031",
    name: "cbam-stability-031",
    suite: "recognition",
    model: "SmallClassifier + CBAM",
    status: "failed",
    date: "2026-11-08 09:24",
    primaryLabel: "Macro-F1",
    primaryValue: 98.49,
    hardware: "RTX 3080 Ti",
    duration: "00m 48s",
    runs: 1,
    weights: {},
    note: "显存碎片导致第三折中断，保留失败上下文。",
  },
];

const state = {
  view: "inspect",
  segmentModel: segmentationModels["small-unet-boundary"],
  recognitionModel: recognitionModels.mobilenetv2,
  threshold: 0.42,
  manifest: null,
  manifestReady: false,
  manifestError: null,
  modelCache: new Map(),
  imageFile: null,
  imageBitmap: null,
  sourceGrayCanvas: null,
  imageObjectUrl: null,
  imageRevision: 0,
  probabilityPreviewUrl: null,
  overlayPreviewUrl: null,
  segProbability: null,
  segGray: null,
  segLatencyMs: 0,
  mask: null,
  analysis: null,
  thresholdTimer: null,
  thresholdRunning: false,
  thresholdQueued: false,
  thresholdRequestId: 0,
  analysisCache: new Map(),
  inferenceStage: "idle",
  inferenceRunning: false,
  trainingRunning: true,
  trainingComplete: false,
  trainingPausedByUser: false,
  trainingProgress: 61.8,
  currentEpoch: 25,
  externalStream: false,
  trainingData: [],
  trainingLogSignature: "",
  uploadTimer: null,
  archiveSearch: "",
  archiveStatus: "all",
  selectedRuns: new Set(["benchmark_v1", "ablation_2026-11-09"]),
  archiveRendered: false,
};

const defaultCode = `from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from steelpipe.models import SmallUNet


def build_optimizer(model: torch.nn.Module, learning_rate: float):
    return torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-4,
    )


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0

    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    return running_loss / len(loader.dataset)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SmallUNet(in_channels=1, out_channels=1).to(device)
    print(f"training on {device}")


if __name__ == "__main__":
    main()`;

class ParticlePipe {
  constructor(container) {
    this.container = container;
    this.scene = null;
    this.camera = null;
    this.renderer = null;
    this.group = null;
    this.surfacePoints = null;
    this.ambientPoints = null;
    this.defectGroup = null;
    this.scanRing = null;
    this.animationFrame = null;
    this.visible = true;
    this.pointer = new THREE.Vector2(0, 0);
    this.pointerTarget = new THREE.Vector2(0, 0);
    this.clock = new THREE.Clock();
    this.init();
  }

  init() {
    try {
      this.scene = new THREE.Scene();
      this.camera = new THREE.PerspectiveCamera(38, 1, 0.1, 100);
      this.camera.position.set(0, 0.15, 8.8);
      this.camera.lookAt(0, 0, 0);

      this.renderer = new THREE.WebGLRenderer({
        antialias: true,
        alpha: true,
        powerPreference: "high-performance",
      });
      this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.8));
      this.renderer.setClearColor(0x000000, 0);
      this.renderer.outputColorSpace = THREE.SRGBColorSpace;
      this.container.appendChild(this.renderer.domElement);

      this.group = new THREE.Group();
      this.group.rotation.z = -0.16;
      this.group.rotation.x = 0.06;
      this.scene.add(this.group);

      this.createStructure();
      this.createAmbientDust();
      this.setDefectsFromAnalysis(null);
      this.resize();

      this.resizeObserver = new ResizeObserver(() => this.resize());
      this.resizeObserver.observe(this.container);
      this.container.addEventListener("pointermove", (event) => this.onPointerMove(event), { passive: true });
      this.container.addEventListener("pointerleave", () => {
        this.pointerTarget.set(0, 0);
      });

      this.loop();
    } catch (error) {
      console.warn("Particle model fallback:", error);
      this.renderFallback();
    }
  }

  createStructure() {
    const positions = [];
    const colors = [];
    const ySteps = 200;
    const angleSteps = 76;
    const baseColor = new THREE.Color("#43d8ce");

    for (let yi = 0; yi <= ySteps; yi += 1) {
      const y = -3.7 + (yi / ySteps) * 7.4;
      for (let ai = 0; ai < angleSteps; ai += 1) {
        const theta = (ai / angleSteps) * Math.PI * 2;
        const wave =
          Math.sin(y * 4.2 + theta * 5.0) * 0.022 +
          Math.sin(y * 12.7 - theta * 2.0) * 0.008 +
          Math.cos(theta * 9.0 + y * 1.8) * 0.012;
        const radius = 1.42 + wave;
        const x = Math.cos(theta) * radius;
        const z = Math.sin(theta) * radius;
        positions.push(x, y, z);

        const intensity = 0.46 + 0.38 * (0.5 + 0.5 * Math.sin(theta * 3 + y * 2.2));
        const color = baseColor.clone().multiplyScalar(intensity);
        colors.push(color.r, color.g, color.b);
      }
    }

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
    geometry.setAttribute("color", new THREE.Float32BufferAttribute(colors, 3));

    const material = new THREE.PointsMaterial({
      size: 0.016,
      vertexColors: true,
      transparent: true,
      opacity: 0.84,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
      sizeAttenuation: true,
    });

    this.surfacePoints = new THREE.Points(geometry, material);
    this.group.add(this.surfacePoints);

    const wireGeometry = new THREE.CylinderGeometry(1.42, 1.42, 7.4, 48, 28, true);
    const wireMaterial = new THREE.MeshBasicMaterial({
      color: 0x2ccbc1,
      wireframe: true,
      transparent: true,
      opacity: 0.035,
      depthWrite: false,
    });
    this.group.add(new THREE.Mesh(wireGeometry, wireMaterial));

    const ringMaterial = new THREE.MeshBasicMaterial({
      color: 0x59f4e4,
      transparent: true,
      opacity: 0.24,
      depthWrite: false,
    });
    [-3.7, 3.7].forEach((y) => {
      const ring = new THREE.Mesh(new THREE.TorusGeometry(1.43, 0.008, 5, 96), ringMaterial.clone());
      ring.rotation.x = Math.PI / 2;
      ring.position.y = y;
      this.group.add(ring);
    });

    this.scanRing = new THREE.Mesh(
      new THREE.TorusGeometry(1.47, 0.014, 6, 96),
      new THREE.MeshBasicMaterial({
        color: 0x59f4e4,
        transparent: true,
        opacity: 0.38,
        depthWrite: false,
        blending: THREE.AdditiveBlending,
      }),
    );
    this.scanRing.rotation.x = Math.PI / 2;
    this.scanRing.position.y = -2.8;
    this.group.add(this.scanRing);
  }

  createAmbientDust() {
    const count = 760;
    const positions = new Float32Array(count * 3);
    const colors = new Float32Array(count * 3);
    const cyan = new THREE.Color("#4fe9dd");

    for (let i = 0; i < count; i += 1) {
      const radius = 2.25 + Math.random() * 4.6;
      const theta = Math.random() * Math.PI * 2;
      const y = (Math.random() - 0.5) * 10.8;
      positions[i * 3] = Math.cos(theta) * radius;
      positions[i * 3 + 1] = y;
      positions[i * 3 + 2] = Math.sin(theta) * radius - 1.2;

      const color = cyan;
      const intensity = 0.18 + Math.random() * 0.34;
      colors[i * 3] = color.r * intensity;
      colors[i * 3 + 1] = color.g * intensity;
      colors[i * 3 + 2] = color.b * intensity;
    }

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));
    const material = new THREE.PointsMaterial({
      size: 0.024,
      vertexColors: true,
      transparent: true,
      opacity: 0.55,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
    });
    this.ambientPoints = new THREE.Points(geometry, material);
    this.scene.add(this.ambientPoints);
  }

  setDefectsFromAnalysis(analysis) {
    if (this.defectGroup) {
      this.group.remove(this.defectGroup);
      this.defectGroup.traverse((child) => {
        child.geometry?.dispose();
        child.material?.dispose();
      });
    }

    this.setScanState(Boolean(analysis?.pixelCount));
    if (!analysis?.mask || !analysis.pixelCount) return;

    const positions = [];
    const colors = [];
    const orange = new THREE.Color("#ff8a3d");
    const warm = new THREE.Color("#ffd0a8");
    const foreground = [];
    for (let index = 0; index < analysis.mask.length; index += 1) {
      if (analysis.mask[index] && (index % 2 === 0 || analysis.probability[index] > 0.75)) {
        foreground.push(index);
      }
    }
    const stride = Math.max(1, Math.ceil(foreground.length / 1250));
    const sampled = foreground.filter((_, index) => index % stride === 0);

    sampled.forEach((index, sampleIndex) => {
      const x = index % analysis.width;
      const imageY = Math.floor(index / analysis.width);
      const theta = (imageY / analysis.height) * Math.PI * 2;
      const y = -3.7 + (x / analysis.width) * 7.4;
      const radius = 1.435 + (Math.random() - 0.5) * 0.04;
      positions.push(Math.cos(theta) * radius, y, Math.sin(theta) * radius);

      const probability = analysis.probability[index];
      const glow = 0.55 + clamp(probability, 0, 1) * 0.45;
      const color = (sampleIndex + x + imageY) % 11 === 0 ? warm : orange;
      colors.push(color.r * glow, color.g * glow, color.b * glow);
    });

    if (!positions.length) return;

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
    geometry.setAttribute("color", new THREE.Float32BufferAttribute(colors, 3));
    const material = new THREE.PointsMaterial({
      size: 0.033,
      vertexColors: true,
      transparent: true,
      opacity: 0.96,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
      sizeAttenuation: true,
    });

    this.defectGroup = new THREE.Points(geometry, material);
    this.group.add(this.defectGroup);
  }

  setScanState(hasDefects) {
    if (!this.scanRing) return;
    this.scanRing.material.color.set(hasDefects ? 0xff8a3d : 0x59f4e4);
    this.scanRing.material.opacity = hasDefects ? 0.58 : 0.28;
  }

  onPointerMove(event) {
    if (prefersReducedMotion.matches) return;
    const rect = this.container.getBoundingClientRect();
    this.pointerTarget.x = ((event.clientX - rect.left) / rect.width - 0.5) * 2;
    this.pointerTarget.y = ((event.clientY - rect.top) / rect.height - 0.5) * 2;
  }

  resize() {
    if (!this.renderer) return;
    const width = Math.max(1, this.container.clientWidth);
    const height = Math.max(1, this.container.clientHeight);
    this.renderer.setSize(width, height, false);
    this.camera.aspect = width / height;
    this.camera.position.z = width < 560 ? 9.7 : 8.8;
    this.camera.updateProjectionMatrix();
  }

  setVisible(visible) {
    this.visible = visible;
    if (visible && !prefersReducedMotion.matches) {
      this.clock.getDelta();
    }
  }

  loop() {
    if (!this.renderer) return;
    this.animationFrame = requestAnimationFrame(() => this.loop());
    if (!this.visible || document.hidden) return;

    const elapsed = this.clock.getElapsedTime();
    this.pointer.lerp(this.pointerTarget, 0.035);

    if (!prefersReducedMotion.matches) {
      this.group.rotation.y = 0.15 + elapsed * 0.075 + this.pointer.x * 0.12;
      this.group.rotation.x = 0.06 - this.pointer.y * 0.055;
      this.group.position.y = Math.sin(elapsed * 0.55) * 0.035;
      this.ambientPoints.rotation.y = -elapsed * 0.012;
      this.scanRing.position.y = -3.55 + ((elapsed * 0.42) % 1) * 7.1;
      this.scanRing.material.opacity =
        (this.defectGroup ? 0.48 : 0.22) + Math.sin(elapsed * 2.1) * 0.12;
      if (this.defectGroup) {
        this.defectGroup.material.opacity = 0.82 + Math.sin(elapsed * 2.7) * 0.1;
      }
    }

    this.renderer.render(this.scene, this.camera);
  }

  renderFallback() {
    this.container.innerHTML = `
      <div class="particle-fallback" role="img" aria-label="钢管截面与缺陷粒子示意">
        <span></span><span></span><span></span>
        <strong>MODEL SURFACE</strong>
      </div>
    `;
  }
}

function init() {
  if (window.lucide) {
    window.lucide.createIcons({
      attrs: { "stroke-width": 1.8 },
    });
  }

  state.trainingData = createInitialTrainingData();
  renderCode(defaultCode, "scripts/train_segmentation.py");
  renderTrainingStages();
  updateTrainingUI();
  updateInferenceUI();
  renderArchive();
  setupNavigation();
  setupReportTabs();
  setupTrainingTabs();
  setupInferenceControls();
  setupTrainingControls();
  setupUploads();
  setupParameterControls();
  setupArchiveControls();
  setupGlobalShortcuts();
  setupTrainingStream();
  updateMobileReportState();

  window.particlePipe = new ParticlePipe($("#particleStage"));
  window.particlePipe.setVisible(state.view === "inspect");
  if (new URLSearchParams(window.location.search).has("debug")) {
    window.__workbenchDebug = {
      getInferenceState: () => ({
        threshold: state.threshold,
        thresholdQueued: state.thresholdQueued,
        thresholdRunning: state.thresholdRunning,
        inferenceRunning: state.inferenceRunning,
        appliedThreshold: state.analysis?.threshold ?? null,
        pixelCount: state.analysis?.pixelCount ?? null,
        segmentationLatencyMs: state.analysis?.segmentationLatencyMs ?? null,
        recognitionLatencyMs: state.analysis?.recognitionLatencyMs ?? null,
        totalLatencyMs: state.analysis?.totalLatencyMs ?? null,
        compressedWasmServed: Boolean(window.__compressedWasmServed),
        cacheKeys: Array.from(state.analysisCache.keys()),
      }),
      countThreshold: (threshold) => {
        if (!state.segProbability) return null;
        let count = 0;
        for (const value of state.segProbability) {
          if (value >= threshold) count += 1;
        }
        return count;
      },
    };
  }
  loadModelManifest();
  startTrainingClock();
}

function setupNavigation() {
  $$(".mode-button").forEach((button) => {
    button.addEventListener("click", () => switchView(button.dataset.view));
  });

  $("#newExperimentBtn").addEventListener("click", () => {
    switchView("train");
    restartTraining();
    showToast("已创建 run-048，数据与代码上下文已继承。");
  });

  const initialHash = window.location.hash.replace("#", "");
  if (["inspect", "train", "archive"].includes(initialHash)) {
    switchView(initialHash, false);
  }

  window.addEventListener("hashchange", () => {
    const next = window.location.hash.replace("#", "");
    if (["inspect", "train", "archive"].includes(next) && next !== state.view) {
      switchView(next, false);
    }
  });
}

function switchView(nextView, updateHash = true) {
  state.view = nextView;
  $$(".workspace-view").forEach((view) => {
    const isActive = view.id === `view-${nextView}`;
    view.hidden = !isActive;
  });
  $$(".mode-button").forEach((button) => {
    const isActive = button.dataset.view === nextView;
    button.classList.toggle("is-active", isActive);
    if (isActive) {
      button.setAttribute("aria-current", "page");
    } else {
      button.removeAttribute("aria-current");
    }
  });

  if (window.particlePipe) {
    window.particlePipe.setVisible(nextView === "inspect");
  }
  if (nextView === "train") {
    updateTrainingUI();
  }
  if (nextView === "archive") {
    renderArchive();
  }
  if (updateHash && window.location.hash !== `#${nextView}`) {
    history.replaceState(null, "", `#${nextView}`);
  }
  if (prefersReducedMotion.matches) {
    window.scrollTo({ top: 0, behavior: "auto" });
  } else {
    window.scrollTo({ top: 0, behavior: "smooth" });
  }
}

function setupReportTabs() {
  setupTabGroup({
    tabSelector: ".report-tab",
    panelSelector: ".report-panel",
    dataAttribute: "reportTab",
    onChange: () => {},
  });

  $("#closeReportBtn").addEventListener("click", closeReport);
  $("#reportToggleBtn").addEventListener("click", () => {
    if ($("#reportDrawer").classList.contains("is-open")) {
      closeReport();
    } else {
      openReport();
    }
  });
}

function setupTrainingTabs() {
  setupTabGroup({
    tabSelector: ".shelf-tab",
    panelSelector: ".shelf-panel",
    dataAttribute: "trainTab",
    onChange: (value) => {
      const labels = {
        data: "数据已锁定 · 按钢管编号分组",
        code: "代码只读预览 · 上传后可替换当前脚本",
        params: "配置修改将在下一次运行时生效",
      };
      $("#shelfState").textContent = labels[value];
    },
  });
}

function setupTabGroup({ tabSelector, panelSelector, dataAttribute, onChange }) {
  const tabs = $$(tabSelector);
  const panels = $$(panelSelector);

  const activate = (nextTab) => {
    const value = nextTab.dataset[dataAttribute];
    tabs.forEach((tab) => {
      const active = tab === nextTab;
      tab.classList.toggle("is-active", active);
      tab.setAttribute("aria-selected", String(active));
      tab.tabIndex = active ? 0 : -1;
    });
    panels.forEach((panel) => {
      panel.hidden = panel.id !== nextTab.getAttribute("aria-controls");
    });
    onChange?.(value);
  };

  tabs.forEach((tab, index) => {
    tab.tabIndex = tab.classList.contains("is-active") ? 0 : -1;
    tab.addEventListener("click", () => activate(tab));
    tab.addEventListener("keydown", (event) => {
      if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
      event.preventDefault();
      let nextIndex = index;
      if (event.key === "ArrowRight") nextIndex = (index + 1) % tabs.length;
      if (event.key === "ArrowLeft") nextIndex = (index - 1 + tabs.length) % tabs.length;
      if (event.key === "Home") nextIndex = 0;
      if (event.key === "End") nextIndex = tabs.length - 1;
      tabs[nextIndex].focus();
      activate(tabs[nextIndex]);
    });
  });
}

function setupInferenceControls() {
  $$(".weight-option[data-seg]").forEach((button) => {
    button.addEventListener("click", async () => {
      if (isInferenceBusy() || !state.manifestReady) return;
      state.segmentModel = segmentationModels[button.dataset.seg];
      $$(".weight-option[data-seg]").forEach((item) => {
        const selected = item === button;
        item.classList.toggle("is-selected", selected);
        item.setAttribute("aria-pressed", String(selected));
      });
      state.analysis = null;
      state.mask = null;
      state.segProbability = null;
      window.particlePipe?.setDefectsFromAnalysis(null);
      updateInferenceUI();
      showToast(`分割权重切换为 ${state.segmentModel.name}。`);
      if (state.imageBitmap) {
        await runInference();
      }
    });
  });

  $$(".weight-option[data-rec]").forEach((button) => {
    button.addEventListener("click", async () => {
      if (isInferenceBusy() || !state.manifestReady) return;
      state.recognitionModel = recognitionModels[button.dataset.rec];
      $$(".weight-option[data-rec]").forEach((item) => {
        const selected = item === button;
        item.classList.toggle("is-selected", selected);
        item.setAttribute("aria-pressed", String(selected));
      });
      updateInferenceUI();
      showToast(`识别权重切换为 ${state.recognitionModel.name}。`);
      if (state.segProbability && state.imageBitmap) {
        scheduleThresholdRecompute(0, true);
      }
    });
  });

  $("#thresholdRange").addEventListener("input", (event) => {
    state.threshold = Number(event.target.value);
    $("#thresholdValue").textContent = state.threshold.toFixed(2);
    $("#imageThreshold").textContent = state.threshold.toFixed(2);
    updateInferenceUI();
    scheduleThresholdRecompute(64);
  });

  $("#thresholdRange").addEventListener("change", () => {
    scheduleThresholdRecompute(0, true);
  });

  $("#runInferenceBtn").addEventListener("click", runInference);
  $("#inferenceImageUpload").addEventListener("change", (event) => {
    const [file] = event.target.files || [];
    if (file) {
      handleInferenceImage(file);
      event.target.value = "";
    }
  });

  const uploadLabel = $('label[for="inferenceImageUpload"]');
  uploadLabel.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      $("#inferenceImageUpload").click();
    }
  });
}

async function loadModelManifest() {
  try {
    const response = await fetch("./models/manifest.json", { cache: "no-cache" });
    if (!response.ok) {
      throw new Error(`模型清单加载失败：HTTP ${response.status}`);
    }
    const manifest = await response.json();
    if (!Array.isArray(manifest.segmentation) || !Array.isArray(manifest.recognition)) {
      throw new Error("模型清单格式无效。");
    }

    manifest.segmentation.forEach((entry) => {
      const model = segmentationModels[entry.id];
      if (model) {
        model.manifest = entry;
        model.name = entry.displayName;
        model.file = entry.file;
      }
    });
    manifest.recognition.forEach((entry) => {
      const model = recognitionModels[entry.id];
      if (model) {
        model.manifest = entry;
        model.name = entry.displayName;
        model.file = entry.file;
      }
    });

    const missing = [
      ...Object.values(segmentationModels),
      ...Object.values(recognitionModels),
    ].filter((model) => !model.file);
    if (missing.length) {
      throw new Error(`模型清单缺少权重：${missing.map((model) => model.id).join(", ")}`);
    }

    state.manifest = manifest;
    state.manifestReady = true;
    updateWeightOptionLabels();
    $("#modelStateBadge").textContent = "REAL ONNX READY";
    $("#weightIntegrityNote").textContent =
      `已加载 ${manifest.segmentation.length + manifest.recognition.length} 个真实 PyTorch 导出权重；浏览器仅执行所选检查点。`;
    setSceneState("ONNX READY");
    syncRunButtonState();
  } catch (error) {
    state.manifestError = error;
    console.error(error);
    $("#modelStateBadge").textContent = "MODEL MANIFEST ERROR";
    $("#weightIntegrityNote").textContent = "真实权重清单加载失败，推理功能已停用。";
    setSceneState("WEIGHT ERROR");
    setReportEmpty("error", "模型清单不可用", "请确认 models/manifest.json 与 ONNX 文件已正确发布。");
    showToast("真实权重清单加载失败，请检查模型文件。", "warning");
    $$(".weight-option").forEach((button) => {
      button.disabled = true;
    });
    syncRunButtonState();
  }
}

function updateWeightOptionLabels() {
  $$(".weight-option[data-seg]").forEach((button) => {
    const model = segmentationModels[button.dataset.seg];
    const em = button.querySelector("em");
    if (model?.manifest?.metrics?.dice != null) {
      em.textContent = `DICE ${(model.manifest.metrics.dice * 100).toFixed(2)}%`;
      button.disabled = false;
    } else {
      em.textContent = "不可用";
      button.disabled = true;
    }
  });
  $$(".weight-option[data-rec]").forEach((button) => {
    const model = recognitionModels[button.dataset.rec];
    const em = button.querySelector("em");
    if (model?.manifest?.metrics?.accuracy != null) {
      em.textContent = `ACC ${(model.manifest.metrics.accuracy * 100).toFixed(2)}%`;
      button.disabled = false;
    } else {
      em.textContent = "不可用";
      button.disabled = true;
    }
  });
}

function updateInferenceUI() {
  const seg = state.segmentModel;
  const rec = state.recognitionModel;
  const analysis = state.analysis;

  $("#explainModel").textContent = rec.name;
  $("#imageThreshold").textContent = state.threshold.toFixed(2);
  $("#reportParams").textContent =
    seg.manifest && rec.manifest
      ? `${(seg.manifest.parametersM + rec.manifest.parametersM).toFixed(3)} M`
      : "—";

  if (!analysis) {
    $("#floatDice").textContent = "—";
    $("#floatLatency").textContent = "—";
    $("#floatDefect").textContent = "0.00";
    $("#modelStateBadge").textContent = state.manifestReady
      ? "REAL ONNX · NO INPUT"
      : state.manifestError
        ? "MODEL MANIFEST ERROR"
        : "LOADING REAL WEIGHTS";
    $("#confidenceRow").hidden = true;
    $("#probabilityList").hidden = true;
    $("#reportMetricGrid").hidden = true;
    updateReportImages(null);
    if (!state.manifestError) {
      setReportEmpty(
        "idle",
        state.manifestReady ? "等待图像输入" : "正在加载真实权重",
        state.manifestReady
          ? "上传后会显示分割概率、缺陷区域和识别类别的真实模型输出。"
          : "首次进入不会执行计算，模型只会在上传图像后按需加载。",
      );
    }
    return;
  }

  const confidence = analysis.confidence ? analysis.confidence * 100 : 0;
  $("#floatDice").textContent = analysis.pixelCount.toLocaleString("zh-CN");
  $("#floatLatency").textContent = analysis.totalLatencyMs.toFixed(0);
  $("#floatDefect").textContent = analysis.defectFraction.toFixed(2);
  $("#reportDice").textContent = analysis.pixelCount.toLocaleString("zh-CN");
  $("#reportIou").textContent = String(analysis.regionCount);
  $("#reportBoundary").textContent =
    `${analysis.segmentationLatencyMs.toFixed(0)} / ` +
    `${analysis.recognitionLatencyMs.toFixed(0)} ms`;
  $("#reportMetricGrid").hidden = false;
  updateReportImages(analysis);

  if (!analysis.pixelCount) {
    $("#modelStateBadge").textContent = "ONNX RESULT · NO DEFECT";
    $("#confidenceRow").hidden = true;
    $("#probabilityList").hidden = true;
    setReportEmpty(
      "no-defect",
      "未检测到超过阈值的缺陷",
      `当前阈值 ${state.threshold.toFixed(2)}，分割掩码为 0 像素，因此未调用识别模型。`,
    );
    return;
  }

  if (!analysis.topClass) {
    $("#modelStateBadge").textContent = "ONNX RESULT · CLASSIFYING";
    $("#confidenceRow").hidden = true;
    $("#probabilityList").hidden = true;
    setReportEmpty(
      "idle",
      "分割完成，正在识别",
      `掩码包含 ${analysis.pixelCount.toLocaleString("zh-CN")} 个缺陷像素，正在生成 20% 外扩裁剪并执行分类模型。`,
    );
    return;
  }

  $("#modelStateBadge").textContent = `ONNX RESULT · ${analysis.topClass.label.toUpperCase()}`;
  setReportEmpty("hidden", "", "");
  $("#confidenceRow").hidden = false;
  $("#probabilityList").hidden = false;
  $("#confidenceValue").textContent = confidence.toFixed(2);
  $("#confidenceGauge .confidence-ring").style.background =
    `conic-gradient(var(--cyan) 0deg ${confidence * 3.6}deg, rgba(89, 244, 228, 0.09) ${confidence * 3.6}deg 360deg)`;
  $("#topClass").textContent = analysis.topClass.label;
  $("#confidenceGauge").setAttribute("aria-label", `分类置信度 ${confidence.toFixed(2)}%`);
  $("#topClass").parentElement.querySelector("p").textContent =
    `分类 softmax 来自 ${rec.name}，输入为掩码外扩 20% 后的真实缺陷裁剪。`;
  renderProbabilityList(analysis.probabilities);
}

function renderProbabilityList(probabilities) {
  const list = $("#probabilityList");
  const topValue = probabilities[0]?.value || 1;
  list.innerHTML = probabilities
    .map((item, index) => {
      const rawPercent = item.value * 100;
      const visualPercent = Math.max(0.8, (item.value / topValue) * 100);
      return `
        <div class="probability-item${index === 0 ? " is-top" : ""}">
          <span class="probability-name">${escapeHtml(item.label)}</span>
          <span class="probability-track" aria-hidden="true">
            <span class="probability-fill" style="width:${visualPercent.toFixed(2)}%"></span>
          </span>
          <span class="probability-value">${rawPercent < 0.01 ? rawPercent.toFixed(3) : rawPercent.toFixed(2)}%</span>
        </div>
      `;
    })
    .join("");
}

function setReportEmpty(mode, title, detail) {
  const empty = $("#reportEmpty");
  if (mode === "hidden") {
    empty.hidden = true;
    return;
  }
  empty.hidden = false;
  empty.classList.toggle("is-no-defect", mode === "no-defect");
  empty.querySelector("strong").textContent = title;
  empty.querySelector("p").textContent = detail;
}

function updateReportImages(analysis) {
  setReportImage(
    "#reportOverlayImage",
    "#overlayPlaceholder",
    analysis?.overlayDataUrl,
    "等待推理",
  );
  setReportImage(
    "#reportMaskImage",
    "#maskPlaceholder",
    analysis?.maskDataUrl,
    "等待推理",
  );
  setReportImage(
    "#recognitionCropImage",
    "#cropPlaceholder",
    analysis?.cropDataUrl,
    analysis?.pixelCount ? "等待识别" : "无缺陷区域",
  );
}

function setReportImage(imageSelector, placeholderSelector, dataUrl, placeholderText) {
  const image = $(imageSelector);
  const placeholder = $(placeholderSelector);
  if (dataUrl) {
    image.src = dataUrl;
    image.hidden = false;
    placeholder.hidden = true;
  } else {
    image.removeAttribute("src");
    image.hidden = true;
    placeholder.hidden = false;
    placeholder.textContent = placeholderText;
  }
}

async function handleInferenceImage(file) {
  if (!file.type.startsWith("image/") && !/\.(png|jpe?g|bmp|tiff?|webp)$/i.test(file.name)) {
    showToast("请选择 PNG、JPEG、BMP、TIFF 或 WebP 图像。", "warning");
    return;
  }
  if (file.size > 25 * 1024 * 1024) {
    showToast("图像超过 25 MB，请先压缩后再上传。", "warning");
    return;
  }

  try {
    const bitmap = await createImageBitmap(file, { imageOrientation: "from-image" }).catch(
      async () => {
        const url = URL.createObjectURL(file);
        const image = new Image();
        image.src = url;
        await image.decode();
        URL.revokeObjectURL(url);
        return image;
      },
    );
    state.imageBitmap?.close?.();
    state.imageBitmap = bitmap;
    state.imageFile = file;
    state.sourceGrayCanvas = createSourceGrayCanvas(bitmap);
    state.imageRevision += 1;
    state.analysisCache.clear();
    state.probabilityPreviewUrl = null;
    state.overlayPreviewUrl = null;
    state.segProbability = null;
    state.mask = null;
    state.analysis = null;
    if (state.imageObjectUrl) URL.revokeObjectURL(state.imageObjectUrl);
    state.imageObjectUrl = URL.createObjectURL(file);

    const preview = $("#uploadedImagePreview");
    preview.src = state.imageObjectUrl;
    preview.alt = `上传图像预览：${file.name}`;
    preview.hidden = false;
    $("#uploadedFileMeta").textContent =
      `${file.name} · ${bitmap.width} × ${bitmap.height} · ${formatBytes(file.size)}`;
    $(".inference-file-context > i, .inference-file-context > svg")?.setAttribute(
      "hidden",
      "",
    );
    syncRunButtonState();
    updateInferenceUI();
    if (state.manifestReady) {
      getModelSession(state.recognitionModel).catch((error) => {
        console.warn("Recognition model prefetch failed:", error);
      });
    }
    await runInference();
  } catch (error) {
    console.error(error);
    showToast("图像读取失败，请更换文件。", "warning");
  }
}

function syncRunButtonState() {
  const runButton = $("#runInferenceBtn");
  if (!runButton) return;
  runButton.disabled =
    !state.manifestReady || !state.imageBitmap || isInferenceBusy();
}

function isInferenceBusy() {
  return state.inferenceRunning || state.thresholdRunning;
}

async function getModelSession(model) {
  if (model.session) return model.session;
  if (!model.manifest?.file) throw new Error(`权重文件不存在：${model.id}`);
  await compressedRuntimeReady;
  const modelBytes = await fetchModelBytes(model.manifest.file);
  model.session = await ort.InferenceSession.create(modelBytes, {
    executionProviders: ["wasm"],
    graphOptimizationLevel: "all",
    logSeverityLevel: 3,
  });
  return model.session;
}

async function prepareCompressedRuntime() {
  if (
    !("serviceWorker" in navigator) ||
    typeof DecompressionStream === "undefined"
  ) {
    return false;
  }
  try {
    await navigator.serviceWorker.register("./compression-sw.js", {
      scope: "./",
    });
    const ready = await Promise.race([
      navigator.serviceWorker.ready.then(() => true),
      new Promise((resolve) => window.setTimeout(() => resolve(false), 2500)),
    ]);
    if (!ready) return false;
    if (!navigator.serviceWorker.controller) {
      await Promise.race([
        new Promise((resolve) => {
          navigator.serviceWorker.addEventListener("controllerchange", resolve, {
            once: true,
          });
        }),
        new Promise((resolve) => window.setTimeout(resolve, 1500)),
      ]);
    }
    return Boolean(navigator.serviceWorker.controller);
  } catch (error) {
    console.warn("Compressed runtime service worker unavailable:", error);
    return false;
  }
}

async function fetchModelBytes(file) {
  const compressedUrl = new URL(`${file}.gz`, window.location.href).href;
  if (typeof DecompressionStream !== "undefined") {
    try {
      const response = await fetch(compressedUrl, { cache: "force-cache" });
      if (response.ok && response.body) {
        const decompressed = response.body.pipeThrough(
          new DecompressionStream("gzip"),
        );
        return await new Response(decompressed).arrayBuffer();
      }
    } catch (error) {
      console.warn("Compressed model unavailable, using ONNX fallback:", error);
    }
  }
  const response = await fetch(new URL(file, window.location.href).href, {
    cache: "force-cache",
  });
  if (!response.ok) {
    throw new Error(`权重下载失败：HTTP ${response.status}`);
  }
  return response.arrayBuffer();
}

async function runInference() {
  if (state.inferenceRunning) return;
  if (!state.manifestReady) {
    showToast("真实权重仍在加载，请稍候。", "warning");
    return;
  }
  if (!state.imageBitmap) {
    showToast("请先上传一张线激光图像。", "warning");
    return;
  }

  state.inferenceRunning = true;
  syncRunButtonState();
  const button = $("#runInferenceBtn");
  button.classList.add("is-running");
  button.setAttribute("aria-busy", "true");
  button.querySelector("span").textContent = "ONNX 推理中";
  $("#inspectStage").setAttribute("aria-busy", "true");
  $("#inspectStage").classList.add("is-scanning");
  openReport();

  try {
    setSceneState(`LOADING ${state.segmentModel.name}`, true);
    const segmentationSession = await getModelSession(state.segmentModel);
    const prepared = prepareSegmentationInput();
    setSceneState("SEGMENTING 320×320", true);
    const segmentationStart = performance.now();
    const output = await segmentationSession.run({
      [segmentationSession.inputNames[0]]: new ort.Tensor(
        "float32",
        prepared.normalized,
        [1, 1, 320, 320],
      ),
    });
    const probability = Float32Array.from(output[segmentationSession.outputNames[0]].data);
    state.segProbability = probability;
    state.segGray = prepared.gray;
    state.probabilityPreviewUrl = createProbabilityDataUrl(probability, 320, 320);
    state.overlayPreviewUrl = createOverlayDataUrl(
      prepared.gray,
      probability,
      320,
      320,
    );
    state.segLatencyMs = performance.now() - segmentationStart;
    await applyThresholdAndRecognize();
  } catch (error) {
    console.error(error);
    state.analysis = null;
    window.particlePipe?.setDefectsFromAnalysis(null);
    updateInferenceUI();
    setReportEmpty(
      "error",
      "真实推理失败",
      error instanceof Error ? error.message : String(error),
    );
    setSceneState("ONNX ERROR");
    showToast("ONNX 推理失败，请检查浏览器控制台。", "warning");
  } finally {
    state.inferenceRunning = false;
    button.classList.remove("is-running");
    button.removeAttribute("aria-busy");
    button.querySelector("span").textContent =
      state.imageBitmap ? "重新计算" : "执行分割与识别";
    $("#inspectStage").removeAttribute("aria-busy");
    $("#inspectStage").classList.remove("is-scanning");
    syncRunButtonState();
    window.setTimeout(drainThresholdQueue, 0);
  }
}

function createSourceGrayCanvas(bitmap) {
  const sourceCanvas = document.createElement("canvas");
  sourceCanvas.width = bitmap.width;
  sourceCanvas.height = bitmap.height;
  const sourceContext = sourceCanvas.getContext("2d", { willReadFrequently: true });
  sourceContext.drawImage(bitmap, 0, 0);
  const sourceData = sourceContext.getImageData(0, 0, bitmap.width, bitmap.height);
  const pixels = sourceData.data;
  for (let pixel = 0; pixel < pixels.length; pixel += 4) {
    const luminance = Math.round(
      (pixels[pixel] * 299 + pixels[pixel + 1] * 587 + pixels[pixel + 2] * 114) / 1000,
    );
    pixels[pixel] = luminance;
    pixels[pixel + 1] = luminance;
    pixels[pixel + 2] = luminance;
  }
  sourceContext.putImageData(sourceData, 0, 0);
  return sourceCanvas;
}

function prepareSegmentationInput() {
  const gray = resizeGrayWindow(
    state.sourceGrayCanvas,
    0,
    0,
    state.sourceGrayCanvas.width,
    state.sourceGrayCanvas.height,
    320,
    320,
  );
  const normalized = new Float32Array(320 * 320);
  for (let index = 0; index < gray.length; index += 1) {
    normalized[index] = (gray[index] - 0.5) / 0.5;
  }
  return { gray, normalized };
}

function scheduleThresholdRecompute(delayMs = 80, immediate = false) {
  if (!state.segProbability || !state.imageBitmap) return;
  state.thresholdQueued = true;
  state.thresholdRequestId += 1;
  window.clearTimeout(state.thresholdTimer);
  state.thresholdTimer = window.setTimeout(
    drainThresholdQueue,
    immediate ? 0 : delayMs,
  );
}

async function drainThresholdQueue() {
  if (!state.thresholdQueued) return;
  if (state.inferenceRunning || state.thresholdRunning) return;
  if (!state.segProbability || !state.imageBitmap) {
    state.thresholdQueued = false;
    return;
  }

  state.thresholdQueued = false;
  state.thresholdRunning = true;
  const requestedThreshold = state.threshold;
  syncRunButtonState();
  const runButton = $("#runInferenceBtn");
  const thresholdSlider = $("#thresholdRange");
  runButton.classList.add("is-running");
  runButton.setAttribute("aria-busy", "true");
  runButton.querySelector("span").textContent = "更新阈值";
  thresholdSlider.setAttribute("aria-busy", "true");
  $("#thresholdHelp").textContent = `正在应用阈值 ${requestedThreshold.toFixed(2)}…`;
  setSceneState("UPDATING THRESHOLD", true);

  try {
    await applyThresholdAndRecognize();
  } catch (error) {
    console.error(error);
    showToast("阈值重算失败，请重试。", "warning");
    setSceneState("THRESHOLD ERROR");
  } finally {
    state.thresholdRunning = false;
    runButton.classList.remove("is-running");
    runButton.removeAttribute("aria-busy");
    runButton.querySelector("span").textContent = "重新计算";
    thresholdSlider.removeAttribute("aria-busy");
    $("#thresholdHelp").textContent = "阈值降低会扩大缺陷响应区域。";
    syncRunButtonState();
    if (
      state.thresholdQueued ||
      Math.abs(state.threshold - requestedThreshold) > 0.0001
    ) {
      state.thresholdQueued = true;
      window.setTimeout(drainThresholdQueue, 0);
    }
  }
}

async function applyThresholdAndRecognize() {
  if (!state.segProbability || !state.imageBitmap) return;
  const threshold = state.threshold;
  const cacheKey = [
    state.imageRevision,
    state.segmentModel.id,
    state.recognitionModel.id,
    threshold.toFixed(2),
  ].join(":");
  const cached = state.analysisCache.get(cacheKey);
  if (cached) {
    state.analysis = cached;
    window.particlePipe?.setDefectsFromAnalysis(cached);
    setSceneState(
      cached.pixelCount
        ? `ONNX RESULT · ${cached.topClass.label.toUpperCase()}`
        : "ONNX RESULT · NO DEFECT",
    );
    updateInferenceUI();
    return;
  }

  const mask = new Uint8Array(state.segProbability.length);
  let pixelCount = 0;
  for (let index = 0; index < mask.length; index += 1) {
    if (state.segProbability[index] >= threshold) {
      mask[index] = 1;
      pixelCount += 1;
    }
  }
  state.mask = mask;

  const morphology = analyzeMask(mask, 320, 320);
  const baseAnalysis = {
    width: 320,
    height: 320,
    threshold,
    probability: state.segProbability,
    mask,
    pixelCount,
    defectFraction: (pixelCount / mask.length) * 100,
    regionCount: morphology.regionCount,
    bounds: morphology.bounds,
    segmentationLatencyMs: state.segLatencyMs || 0,
    recognitionLatencyMs: 0,
    totalLatencyMs: state.segLatencyMs || 0,
    probabilities: [],
    topClass: null,
    confidence: 0,
    probabilityDataUrl: state.probabilityPreviewUrl,
    maskDataUrl: createMaskDataUrl(mask, 320, 320),
    overlayDataUrl: state.overlayPreviewUrl,
    cropDataUrl: null,
  };
  state.analysis = baseAnalysis;
  updateInferenceUI();

  if (!pixelCount) {
    window.particlePipe?.setDefectsFromAnalysis(baseAnalysis);
    setSceneState("ONNX RESULT · NO DEFECT");
    cacheAnalysis(cacheKey, baseAnalysis);
    if (!state.thresholdRunning) {
      showToast(`阈值 ${threshold.toFixed(2)} 下未检测到缺陷，识别模型未运行。`);
    }
    return;
  }

  const recognitionSessionPromise = getModelSession(state.recognitionModel);
  setSceneState(`LOADING ${state.recognitionModel.name}`, true);
  const recognitionSession = await recognitionSessionPromise;
  const recognitionInput = prepareRecognitionInput(mask, baseAnalysis.bounds);
  setSceneState("CLASSIFYING DEFECT CROP", true);
  const recognitionStart = performance.now();
  const output = await recognitionSession.run({
    [recognitionSession.inputNames[0]]: new ort.Tensor(
      "float32",
      recognitionInput.input,
      [1, 2, 128, 128],
    ),
  });
  const recognitionLatency = performance.now() - recognitionStart;
  const values = Float32Array.from(output[recognitionSession.outputNames[0]].data);
  const classes = state.recognitionModel.manifest.classes;
  const probabilities = classes
    .map((label, index) => ({ label, value: Number(values[index]) || 0 }))
    .sort((left, right) => right.value - left.value);
  const topClass = probabilities[0];
  const totalLatencyMs =
    (state.segLatencyMs || 0) + recognitionLatency;

  state.analysis = {
    ...baseAnalysis,
    probabilities,
    topClass,
    confidence: topClass.value,
    recognitionLatencyMs: recognitionLatency,
    totalLatencyMs,
    cropDataUrl: recognitionInput.cropDataUrl,
  };
  window.particlePipe?.setDefectsFromAnalysis(state.analysis);
  const completedAnalysis = state.analysis;
  cacheAnalysis(cacheKey, completedAnalysis);
  setSceneState(`ONNX RESULT · ${topClass.label.toUpperCase()}`);
  updateInferenceUI();
  if (!state.thresholdRunning) {
    showToast(
      `真实推理完成：${topClass.label} ${(topClass.value * 100).toFixed(2)}%，阈值 ${threshold.toFixed(2)}。`,
    );
  }
}

function cacheAnalysis(key, analysis) {
  state.analysisCache.delete(key);
  state.analysisCache.set(key, analysis);
  while (state.analysisCache.size > 16) {
    state.analysisCache.delete(state.analysisCache.keys().next().value);
  }
}

function analyzeMask(mask, width, height) {
  const visited = new Uint8Array(mask.length);
  const pending = new Int32Array(mask.length);
  let regionCount = 0;
  let largestRegion = 0;
  let minX = width;
  let minY = height;
  let maxX = -1;
  let maxY = -1;

  for (let index = 0; index < mask.length; index += 1) {
    if (!mask[index]) continue;
    const x = index % width;
    const y = Math.floor(index / width);
    minX = Math.min(minX, x);
    maxX = Math.max(maxX, x);
    minY = Math.min(minY, y);
    maxY = Math.max(maxY, y);
    if (visited[index]) continue;

    visited[index] = 1;
    pending[0] = index;
    let pendingCount = 1;
    let area = 0;
    while (pendingCount) {
      const current = pending[--pendingCount];
      area += 1;
      const currentX = current % width;
      const currentY = Math.floor(current / width);
      if (currentX > 0) {
        const neighbor = current - 1;
        if (mask[neighbor] && !visited[neighbor]) {
          visited[neighbor] = 1;
          pending[pendingCount++] = neighbor;
        }
      }
      if (currentX + 1 < width) {
        const neighbor = current + 1;
        if (mask[neighbor] && !visited[neighbor]) {
          visited[neighbor] = 1;
          pending[pendingCount++] = neighbor;
        }
      }
      if (currentY > 0) {
        const neighbor = current - width;
        if (mask[neighbor] && !visited[neighbor]) {
          visited[neighbor] = 1;
          pending[pendingCount++] = neighbor;
        }
      }
      if (currentY + 1 < height) {
        const neighbor = current + width;
        if (mask[neighbor] && !visited[neighbor]) {
          visited[neighbor] = 1;
          pending[pendingCount++] = neighbor;
        }
      }
    }
    if (area >= 12) regionCount += 1;
    largestRegion = Math.max(largestRegion, area);
  }

  return {
    regionCount: regionCount || (maxX >= minX ? 1 : 0),
    largestRegion,
    bounds:
      maxX >= minX
        ? { x0: minX, y0: minY, x1: maxX + 1, y1: maxY + 1 }
        : null,
  };
}

function prepareRecognitionInput(mask, bounds) {
  const sourceWidth = state.sourceGrayCanvas.width;
  const sourceHeight = state.sourceGrayCanvas.height;
  const scaleX = sourceWidth / 320;
  const scaleY = sourceHeight / 320;
  let x0 = bounds.x0 * scaleX;
  let y0 = bounds.y0 * scaleY;
  let x1 = bounds.x1 * scaleX;
  let y1 = bounds.y1 * scaleY;
  const paddingX = Math.max(1, Math.round((x1 - x0) * 0.2));
  const paddingY = Math.max(1, Math.round((y1 - y0) * 0.2));
  x0 = clamp(Math.floor(x0 - paddingX), 0, sourceWidth - 1);
  y0 = clamp(Math.floor(y0 - paddingY), 0, sourceHeight - 1);
  x1 = clamp(Math.ceil(x1 + paddingX), x0 + 1, sourceWidth);
  y1 = clamp(Math.ceil(y1 + paddingY), y0 + 1, sourceHeight);

  const cropWidth = x1 - x0;
  const cropHeight = y1 - y0;
  const cropGray = resizeGrayWindow(
    state.sourceGrayCanvas,
    x0,
    y0,
    cropWidth,
    cropHeight,
    128,
    128,
  );

  const mask128 = new Float32Array(128 * 128);
  const input = new Float32Array(128 * 128 * 2);
  for (let y = 0; y < 128; y += 1) {
    for (let x = 0; x < 128; x += 1) {
      const index = y * 128 + x;
      input[index] = (cropGray[index] - 0.5) / 0.5;

      const sourceX = clamp(
        Math.floor(x0 + ((x + 0.5) / 128) * cropWidth),
        0,
        sourceWidth - 1,
      );
      const sourceY = clamp(
        Math.floor(y0 + ((y + 0.5) / 128) * cropHeight),
        0,
        sourceHeight - 1,
      );
      const maskX = clamp(Math.floor((sourceX / sourceWidth) * 320), 0, 319);
      const maskY = clamp(Math.floor((sourceY / sourceHeight) * 320), 0, 319);
      mask128[index] = mask[maskY * 320 + maskX] ? 1 : 0;
      input[128 * 128 + index] = mask128[index];
    }
  }

  const previewCanvas = document.createElement("canvas");
  previewCanvas.width = 128;
  previewCanvas.height = 128;
  const previewContext = previewCanvas.getContext("2d");
  const previewImageData = previewContext.createImageData(128, 128);
  for (let index = 0; index < cropGray.length; index += 1) {
    const pixel = index * 4;
    const base = Math.round(cropGray[index] * 255);
    if (mask128[index]) {
      previewImageData.data[pixel] = Math.min(255, base + 92);
      previewImageData.data[pixel + 1] = Math.round(base * 0.72);
      previewImageData.data[pixel + 2] = Math.max(0, base - 35);
    } else {
      previewImageData.data[pixel] = base;
      previewImageData.data[pixel + 1] = base;
      previewImageData.data[pixel + 2] = base;
    }
    previewImageData.data[pixel + 3] = 255;
  }
  previewContext.putImageData(previewImageData, 0, 0);

  return {
    input,
    cropDataUrl: previewCanvas.toDataURL("image/png"),
  };
}

function resizeGrayWindow(
  sourceCanvas,
  sourceX,
  sourceY,
  sourceWidth,
  sourceHeight,
  targetWidth,
  targetHeight,
) {
  const sourceContext = sourceCanvas.getContext("2d", { willReadFrequently: true });
  const sourcePixels = sourceContext.getImageData(
    0,
    0,
    sourceCanvas.width,
    sourceCanvas.height,
  ).data;
  const output = new Float32Array(targetWidth * targetHeight);
  const scaleX = sourceWidth / targetWidth;
  const scaleY = sourceHeight / targetHeight;

  for (let y = 0; y < targetHeight; y += 1) {
    const sourceCenterY = (y + 0.5) * scaleY - 0.5;
    const y0 = Math.floor(sourceCenterY);
    const y1 = y0 + 1;
    const weightY = sourceCenterY - y0;
    const clampedY0 = clamp(sourceY + y0, 0, sourceCanvas.height - 1);
    const clampedY1 = clamp(sourceY + y1, 0, sourceCanvas.height - 1);

    for (let x = 0; x < targetWidth; x += 1) {
      const sourceCenterX = (x + 0.5) * scaleX - 0.5;
      const x0 = Math.floor(sourceCenterX);
      const x1 = x0 + 1;
      const weightX = sourceCenterX - x0;
      const clampedX0 = clamp(sourceX + x0, 0, sourceCanvas.width - 1);
      const clampedX1 = clamp(sourceX + x1, 0, sourceCanvas.width - 1);
      const topLeft = sourcePixels[(clampedY0 * sourceCanvas.width + clampedX0) * 4];
      const topRight = sourcePixels[(clampedY0 * sourceCanvas.width + clampedX1) * 4];
      const bottomLeft = sourcePixels[(clampedY1 * sourceCanvas.width + clampedX0) * 4];
      const bottomRight = sourcePixels[(clampedY1 * sourceCanvas.width + clampedX1) * 4];
      const top = topLeft * (1 - weightX) + topRight * weightX;
      const bottom = bottomLeft * (1 - weightX) + bottomRight * weightX;
      output[y * targetWidth + x] = (top * (1 - weightY) + bottom * weightY) / 255;
    }
  }
  return output;
}

function createProbabilityDataUrl(probability, width, height) {
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext("2d");
  const imageData = context.createImageData(width, height);
  for (let index = 0; index < probability.length; index += 1) {
    const value = Math.round(clamp(probability[index], 0, 1) * 255);
    const pixel = index * 4;
    imageData.data[pixel] = value;
    imageData.data[pixel + 1] = value;
    imageData.data[pixel + 2] = value;
    imageData.data[pixel + 3] = 255;
  }
  context.putImageData(imageData, 0, 0);
  return canvas.toDataURL("image/png");
}

function createMaskDataUrl(mask, width, height) {
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext("2d");
  const imageData = context.createImageData(width, height);
  for (let index = 0; index < mask.length; index += 1) {
    const value = mask[index] ? 255 : 0;
    const pixel = index * 4;
    imageData.data[pixel] = value;
    imageData.data[pixel + 1] = value;
    imageData.data[pixel + 2] = value;
    imageData.data[pixel + 3] = 255;
  }
  context.putImageData(imageData, 0, 0);
  return canvas.toDataURL("image/png");
}

function createOverlayDataUrl(gray, probability, width, height) {
  let minimum = Infinity;
  let maximum = -Infinity;
  for (const value of probability) {
    minimum = Math.min(minimum, value);
    maximum = Math.max(maximum, value);
  }
  const range = maximum - minimum;
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext("2d");
  const imageData = context.createImageData(width, height);
  for (let index = 0; index < gray.length; index += 1) {
    const normalized = range > 1e-8 ? (probability[index] - minimum) / range : 0;
    const red = clamp(2 * normalized, 0, 1);
    const blue = clamp(2 * (1 - normalized), 0, 1);
    const green = clamp(1.5 - Math.abs(normalized - 0.5) * 3, 0, 1);
    const base = clamp(gray[index], 0, 1);
    const pixel = index * 4;
    imageData.data[pixel] = Math.round((base * 0.55 + red * 0.45) * 255);
    imageData.data[pixel + 1] = Math.round((base * 0.55 + green * 0.45) * 255);
    imageData.data[pixel + 2] = Math.round((base * 0.55 + blue * 0.45) * 255);
    imageData.data[pixel + 3] = 255;
  }
  context.putImageData(imageData, 0, 0);
  return canvas.toDataURL("image/png");
}

function setSceneState(text, processing = false) {
  const sceneState = $("#runtimeSceneState");
  sceneState.classList.toggle("is-processing", processing);
  sceneState.innerHTML = `<span></span>${escapeHtml(text)}`;
}

function openReport() {
  const drawer = $("#reportDrawer");
  drawer.classList.add("is-open");
  if (window.matchMedia("(max-width: 900px)").matches) {
    document.body.dataset.reportOpen = "true";
  }
}

function closeReport() {
  $("#reportDrawer").classList.remove("is-open");
  delete document.body.dataset.reportOpen;
}

function updateMobileReportState() {
  if (window.matchMedia("(max-width: 900px)").matches) {
    closeReport();
  } else {
    openReport();
  }
}

function createInitialTrainingData() {
  return Array.from({ length: 24 }, (_, index) => {
    const t = index / 23;
    return {
      loss: 0.284 * Math.exp(-3.15 * t) + 0.036 + Math.sin(index * 1.9) * 0.008,
      dice: 0.704 + 0.17 * (1 - Math.exp(-3.4 * t)) + Math.sin(index * 1.37) * 0.0024,
    };
  });
}

function renderTrainingStages() {
  const rail = $("#trainingStages");
  rail.innerHTML = trainingStages
    .map(
      (stage, index) =>
        `<div class="stage-node" data-stage="${stage.id}" data-index="${index}">
          <span class="stage-dot">${stage.short}</span>
          <span>${stage.label}</span>
        </div>`,
    )
    .join("");
}

function getStageIndex(progress) {
  if (progress >= 100) return 5;
  if (progress >= 93) return 4;
  if (progress >= 85) return 3;
  if (progress >= 12) return 2;
  if (progress >= 5) return 1;
  return 0;
}

function updateTrainingUI() {
  const progress = Math.min(100, state.trainingProgress);
  $("#trainingProgressValue").textContent = progress.toFixed(1);
  $("#liveLoss").textContent = state.trainingData.at(-1)?.loss.toFixed(4) || "0.0000";
  $("#liveDice").textContent = state.trainingData.at(-1)?.dice.toFixed(4) || "0.0000";
  $("#liveEpoch").textContent = String(state.currentEpoch).padStart(2, "0");

  const remainingMinutes = Math.max(0, (100 - progress) * 0.315);
  const minutes = Math.floor(remainingMinutes);
  const seconds = Math.round((remainingMinutes - minutes) * 60);
  $("#trainingEta").textContent =
    progress >= 100 ? "训练完成 · 检查点已归档" : `预计 ${minutes} 分 ${String(seconds).padStart(2, "0")} 秒完成`;

  const statusBadge = $("#trainingStatusBadge");
  if (state.trainingComplete) {
    statusBadge.className = "run-state";
    statusBadge.innerHTML = "<span></span>COMPLETE";
  } else if (!state.trainingRunning) {
    statusBadge.className = "run-state is-paused";
    statusBadge.innerHTML = "<span></span>PAUSED";
  } else {
    statusBadge.className = "run-state is-live";
    statusBadge.innerHTML = "<span></span>RUNNING";
  }

  const activeStage = getStageIndex(progress);
  $$(".stage-node").forEach((node, index) => {
    node.classList.toggle("is-complete", index < activeStage || progress >= 100);
    node.classList.toggle("is-active", index === activeStage && progress < 100);
    const dot = node.querySelector(".stage-dot");
    dot.textContent = index < activeStage || progress >= 100 ? "✓" : trainingStages[index].short;
  });

  const pauseButton = $("#pauseTrainingBtn");
  if (pauseButton) {
    const label = pauseButton.querySelector("span");
    const icon = pauseButton.querySelector("i, svg");
    let nextIcon = "pause";
    if (state.trainingComplete) {
      pauseButton.disabled = true;
      label.textContent = "已完成";
      nextIcon = "check";
    } else if (state.trainingRunning) {
      pauseButton.disabled = false;
      label.textContent = "暂停";
      nextIcon = "pause";
    } else {
      pauseButton.disabled = false;
      label.textContent = "继续";
      nextIcon = "play";
    }
    if (icon) {
      const replacement = document.createElement("i");
      replacement.setAttribute("data-lucide", nextIcon);
      replacement.setAttribute("aria-hidden", "true");
      icon.replaceWith(replacement);
    }
    window.lucide?.createIcons({ attrs: { "stroke-width": 1.8 } });
  }

  renderTrainingChart();
  updateTrainingVitals();
  updateEventLogThreshold();
}

function renderTrainingChart() {
  const chart = $("#trainingChart");
  if (!chart) return;
  const rect = chart.getBoundingClientRect();
  const width = Math.max(280, Math.round(rect.width || 760));
  const height = Math.max(190, Math.round(rect.height || 292));
  chart.setAttribute("viewBox", `0 0 ${width} ${height}`);
  const left = width < 500 ? 32 : 44;
  const right = width - (width < 500 ? 10 : 26);
  const top = width < 500 ? 14 : 20;
  const bottom = height - (width < 500 ? 38 : 46);
  const data = state.trainingData;
  if (!data.length) return;

  const xAt = (index) => left + (index / Math.max(1, data.length - 1)) * (right - left);
  const lossY = (value) => bottom - ((clamp(value, 0.02, 0.33) - 0.02) / 0.31) * (bottom - top);
  const diceY = (value) => bottom - ((clamp(value, 0.68, 0.91) - 0.68) / 0.23) * (bottom - top);

  const lossPath = data.map((point, index) => `${index === 0 ? "M" : "L"} ${xAt(index).toFixed(2)} ${lossY(point.loss).toFixed(2)}`).join(" ");
  const dicePath = data.map((point, index) => `${index === 0 ? "M" : "L"} ${xAt(index).toFixed(2)} ${diceY(point.dice).toFixed(2)}`).join(" ");
  const areaPath = `${lossPath} L ${xAt(data.length - 1).toFixed(2)} ${bottom} L ${left} ${bottom} Z`;

  $("#lossLine").setAttribute("d", lossPath);
  $("#diceLine").setAttribute("d", dicePath);
  $("#lossArea").setAttribute("d", areaPath);

  const lastIndex = data.length - 1;
  const last = data[lastIndex];
  $("#lossCursor").setAttribute("cx", xAt(lastIndex).toFixed(2));
  $("#lossCursor").setAttribute("cy", lossY(last.loss).toFixed(2));
  $("#diceCursor").setAttribute("cx", xAt(lastIndex).toFixed(2));
  $("#diceCursor").setAttribute("cy", diceY(last.dice).toFixed(2));

  const grid = $(".chart-grid", chart);
  const axis = $(".chart-axis", chart);
  const gridLines = [0, 1, 2, 3, 4]
    .map((index) => {
      const y = top + (index / 4) * (bottom - top);
      return `<line x1="${left}" y1="${y}" x2="${right}" y2="${y}" />`;
    })
    .join("");
  grid.innerHTML = gridLines;

  const epochTotal = Math.max(state.currentEpoch, 24);
  const xTicks = [
    { x: left, label: "01" },
    { x: left + (right - left) * 0.33, label: String(Math.round(epochTotal * 0.33)).padStart(2, "0") },
    { x: left + (right - left) * 0.66, label: String(Math.round(epochTotal * 0.66)).padStart(2, "0") },
    { x: right, label: String(epochTotal).padStart(2, "0") },
  ];

  const xLabels = xTicks
    .map((tick) => `<text x="${tick.x}" y="${height - 18}" text-anchor="middle">${tick.label}</text>`)
    .join("");
  const yLabels = ["0.33", "0.25", "0.17", "0.09", "0.02"]
    .map((label, index) => {
      const y = top + (index / 4) * (bottom - top);
      return `<text x="${width < 500 ? 2 : 8}" y="${y + 3}">${label}</text>`;
    })
    .join("");
  axis.innerHTML = `${xLabels}${yLabels}<text x="${(left + right) / 2}" y="${height - 2}" text-anchor="middle">EPOCH / SAMPLE</text>`;
}

function updateTrainingVitals() {
  if (!state.trainingRunning) return;
  const base = state.trainingProgress;
  const gpu = Math.round(clamp(74 + Math.sin(base * 0.12) * 7 + Math.random() * 5, 61, 94));
  const vram = clamp(17.8 + Math.sin(base * 0.05) * 0.9 + Math.random() * 0.45, 17.2, 20.4);
  const throughput = Math.round(clamp(120 + Math.sin(base * 0.2) * 11 + Math.random() * 7, 108, 142));
  const temperature = Math.round(clamp(58 + gpu * 0.06 + Math.random() * 2, 60, 69));
  const loader = Math.round(clamp(81 + Math.sin(base * 0.16) * 5 + Math.random() * 4, 74, 92));
  const bandwidth = Math.round(clamp(58 + Math.cos(base * 0.13) * 7 + Math.random() * 5, 48, 72));

  $("#gpuLoad").textContent = gpu;
  $("#gpuRing").style.background =
    `conic-gradient(var(--cyan) 0deg ${gpu * 3.6}deg, rgba(89, 244, 228, 0.08) ${gpu * 3.6}deg 360deg)`;
  $("#vramValue").textContent = `${vram.toFixed(1)} / 80 GB`;
  $("#throughputValue").textContent = `${throughput} img/s`;
  $("#temperatureValue").textContent = `${temperature} °C`;
  $("#loaderValue").textContent = `${loader}%`;
  $("#loaderBar").style.width = `${loader}%`;
  $("#bandwidthValue").textContent = `${bandwidth}%`;
  $("#bandwidthBar").style.width = `${bandwidth}%`;
}

function startTrainingClock() {
  if (state.trainingTimer) return;
  state.trainingTimer = window.setInterval(() => {
    if (state.externalStream) return;
    if (!state.trainingRunning) return;
    const increment = 0.34 + Math.random() * 0.52;
    state.trainingProgress = Math.min(100, state.trainingProgress + increment);
    state.currentEpoch = Math.min(40, Math.floor(state.trainingProgress / 2.45) + 1);

    const t = state.currentEpoch / 40;
    const loss = 0.286 * Math.exp(-3.2 * t) + 0.032 + (Math.random() - 0.5) * 0.004;
    const dice = 0.704 + 0.171 * (1 - Math.exp(-3.45 * t)) + (Math.random() - 0.5) * 0.0018;
    state.trainingData.push({ loss, dice });
    if (state.trainingData.length > 24) state.trainingData.shift();

    if (state.trainingProgress >= 100) {
      state.trainingProgress = 100;
      state.currentEpoch = 40;
      state.trainingComplete = true;
      state.trainingRunning = false;
      addTrainingEvent("ok", "checkpoint saved: runs/047/best.pt");
      showToast("run-047 训练完成，验证 Dice 0.8755，检查点已归档。");
    } else if (Math.random() > 0.57) {
      const messages = [
        "val step complete · dice stable",
        "AMP scaler updated · no overflow",
        "dataloader prefetch 84%",
        "gradient norm within range",
        "checkpoint shard verified",
      ];
      addTrainingEvent("ok", messages[Math.floor(Math.random() * messages.length)]);
    }

    updateTrainingUI();
  }, 1200);
}

function setupTrainingStream() {
  const streamUrl = new URLSearchParams(window.location.search).get("stream");
  if (!streamUrl) return;

  let eventSource;
  try {
    eventSource = new EventSource(streamUrl);
  } catch {
    showToast("训练流地址无效，已切换为本地演示数据。", "warning");
    return;
  }

  state.externalStream = true;
  let warned = false;

  const applySnapshot = (event) => {
    let payload;
    try {
      payload = JSON.parse(event.data);
    } catch {
      return;
    }
    if (Number.isFinite(payload.progress)) {
      state.trainingProgress = clamp(payload.progress, 0, 100);
    }
    if (Number.isFinite(payload.epoch)) {
      state.currentEpoch = Math.max(1, Math.round(payload.epoch));
    }
    if (Number.isFinite(payload.loss) || Number.isFinite(payload.dice)) {
      state.trainingData.push({
        loss: Number.isFinite(payload.loss) ? payload.loss : state.trainingData.at(-1)?.loss || 0.2,
        dice: Number.isFinite(payload.dice) ? payload.dice : state.trainingData.at(-1)?.dice || 0.71,
      });
      if (state.trainingData.length > 24) state.trainingData.shift();
    }
    if (payload.status === "completed" || state.trainingProgress >= 100) {
      state.trainingComplete = true;
      state.trainingRunning = false;
    } else if (payload.status === "paused") {
      state.trainingRunning = false;
    } else if (payload.status === "running") {
      state.trainingRunning = true;
    }
    updateTrainingUI();
  };

  eventSource.onmessage = applySnapshot;
  eventSource.addEventListener("snapshot", applySnapshot);
  eventSource.addEventListener("log", (event) => {
    try {
      const payload = JSON.parse(event.data);
      addTrainingEvent(payload.level === "warn" ? "warn" : "ok", payload.message || String(event.data));
    } catch {
      addTrainingEvent("ok", String(event.data));
    }
  });
  eventSource.onerror = () => {
    state.externalStream = false;
    eventSource.close();
    if (!warned) {
      warned = true;
      showToast("训练流暂时断开，工作台已回退到本地进度演示。", "warning");
    }
  };
}

function setupTrainingControls() {
  $("#pauseTrainingBtn").addEventListener("click", () => {
    if (state.trainingComplete) return;
    state.trainingRunning = !state.trainingRunning;
    state.trainingPausedByUser = !state.trainingRunning;
    addTrainingEvent(
      state.trainingRunning ? "ok" : "warn",
      state.trainingRunning ? "run resumed by operator" : "run paused by operator",
    );
    updateTrainingUI();
    showToast(state.trainingRunning ? "训练已继续。" : "训练已暂停，检查点保持安全。");
  });

  $("#restartTrainingBtn").addEventListener("click", restartTraining);
}

function restartTraining() {
  state.trainingProgress = 0;
  state.currentEpoch = 1;
  state.trainingRunning = true;
  state.trainingComplete = false;
  state.trainingPausedByUser = false;
  state.trainingData = createInitialTrainingData().slice(0, 2);
  state.trainingLogSignature = "";
  $("#eventLog").innerHTML = "";
  addTrainingEvent("ok", "run-048 initialized · RTX 4090D");
  addTrainingEvent("ok", "dataset lock verified · 12 pipes");
  addTrainingEvent("ok", "SmallUNet + boundary loss compiled");
  updateTrainingUI();
  showToast("新训练运行已启动，正在装载数据。");
}

function addTrainingEvent(type, message) {
  const log = $("#eventLog");
  if (!log) return;
  const now = new Date();
  const time = now.toLocaleTimeString("zh-CN", {
    hour12: false,
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
  const line = document.createElement("div");
  line.className = `event-line${type === "warn" ? " is-warn" : ""}`;
  line.innerHTML = `<time>${time}</time><strong>${type.toUpperCase()}</strong><span>${escapeHtml(message)}</span>`;
  log.prepend(line);
  while (log.children.length > 4) {
    log.lastElementChild.remove();
  }
}

function updateEventLogThreshold() {
  if (!$("#eventLog").children.length) {
    addTrainingEvent("ok", "run-047 resumed · epoch 25");
    addTrainingEvent("ok", "validation dice 0.8718");
    addTrainingEvent("ok", "checkpoint epoch_024.pt verified");
  }
  const stage = getStageIndex(state.trainingProgress);
  const signature = `${stage}-${state.currentEpoch}`;
  if (signature !== state.trainingLogSignature && state.trainingRunning) {
    state.trainingLogSignature = signature;
  }
}

function setupUploads() {
  const datasetInput = $("#datasetUpload");
  const datasetDropzone = $("#datasetDropzone");

  datasetInput.addEventListener("change", () => {
    if (datasetInput.files?.length) processDatasetUpload(datasetInput.files);
  });

  ["dragenter", "dragover"].forEach((eventName) => {
    datasetDropzone.addEventListener(eventName, (event) => {
      event.preventDefault();
      datasetDropzone.classList.add("is-dragging");
    });
  });

  ["dragleave", "drop"].forEach((eventName) => {
    datasetDropzone.addEventListener(eventName, (event) => {
      event.preventDefault();
      datasetDropzone.classList.remove("is-dragging");
    });
  });

  datasetDropzone.addEventListener("drop", (event) => {
    const files = event.dataTransfer?.files;
    if (files?.length) processDatasetUpload(files);
  });

  datasetDropzone.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      datasetInput.click();
    }
  });
  datasetDropzone.tabIndex = 0;

  $("#codeUpload").addEventListener("change", (event) => {
    const file = event.target.files?.[0];
    if (!file) return;
    if (file.size > 250000) {
      showToast("代码文件超过 250 KB，已拒绝预览。", "warning");
      return;
    }
    const reader = new FileReader();
    reader.onload = () => {
      renderCode(String(reader.result || ""), file.name);
      showToast(`代码预览已加载：${file.name}`);
    };
    reader.onerror = () => showToast("代码文件读取失败，请重试。", "warning");
    reader.readAsText(file.slice(0, 250000));
  });

  const codeUploadLabel = $('label[for="codeUpload"]');
  codeUploadLabel.tabIndex = 0;
  codeUploadLabel.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      $("#codeUpload").click();
    }
  });

  $("#resetCodeBtn").addEventListener("click", () => {
    $("#codeUpload").value = "";
    renderCode(defaultCode, "scripts/train_segmentation.py");
    showToast("已恢复仓库中的训练脚本示例。");
  });
}

function processDatasetUpload(fileList) {
  const files = Array.from(fileList);
  if (!files.length) return;
  if (state.uploadTimer) window.clearInterval(state.uploadTimer);

  const totalBytes = files.reduce((sum, file) => sum + file.size, 0);
  $("#uploadProgress").hidden = false;
  $("#uploadFileName").textContent =
    files.length === 1 ? files[0].name : `${files[0].name} + ${files.length - 1} 个文件`;
  $("#uploadPercent").textContent = "0%";
  $("#uploadBar").style.width = "0%";
  let progress = 0;

  state.uploadTimer = window.setInterval(() => {
    progress = Math.min(100, progress + 7 + Math.random() * 11);
    $("#uploadPercent").textContent = `${Math.round(progress)}%`;
    $("#uploadBar").style.width = `${progress}%`;
    $("#uploadFileName").textContent =
      progress < 42
        ? "正在读取文件与校验尺寸"
        : progress < 78
          ? "正在校验 gray / masks 配对关系"
          : "正在建立按钢管编号分组的 manifest";

    if (progress >= 100) {
      window.clearInterval(state.uploadTimer);
      state.uploadTimer = null;
      $("#uploadFileName").textContent = `上传完成 · ${formatBytes(totalBytes)}`;
      $("#datasetFileStrip").innerHTML = files
        .slice(0, 5)
        .map(
          (file) =>
            `<span><i data-lucide="file-check-2" aria-hidden="true"></i>${escapeHtml(file.name)}</span>`,
        )
        .join("");
      window.lucide?.createIcons({ attrs: { "stroke-width": 1.8 } });
      showToast(`数据集校验完成：${files.length} 个文件，${formatBytes(totalBytes)}。`);
    }
  }, 180);
}

function setupParameterControls() {
  const controls = [
    {
      input: $("#learningRate"),
      output: $("#learningRateValue"),
      format: (value) => Number(value).toFixed(5),
    },
    {
      input: $("#batchSize"),
      output: $("#batchSizeValue"),
      format: (value) => String(value),
    },
    {
      input: $("#epochCount"),
      output: $("#epochCountValue"),
      format: (value) => String(value),
    },
    {
      input: $("#patience"),
      output: $("#patienceValue"),
      format: (value) => String(value),
    },
  ];

  controls.forEach(({ input, output, format }) => {
    input.addEventListener("input", () => {
      output.textContent = format(input.value);
    });
  });
}

function setupArchiveControls() {
  $("#archiveSearch").addEventListener("input", (event) => {
    state.archiveSearch = event.target.value.trim().toLowerCase();
    renderArchive();
  });

  $$(".filter-chip").forEach((button) => {
    button.addEventListener("click", () => {
      state.archiveStatus = button.dataset.status;
      $$(".filter-chip").forEach((item) => {
        const active = item === button;
        item.classList.toggle("is-active", active);
        item.setAttribute("aria-pressed", String(active));
      });
      renderArchive();
    });
  });

  $("#clearCompareBtn").addEventListener("click", () => {
    state.selectedRuns.clear();
    renderArchive();
    showToast("已清空对比选择。");
  });

  $("#openCompareBtn").addEventListener("click", () => {
    if (state.selectedRuns.size < 2) {
      showToast("至少选择两条运行记录后再比较。", "warning");
      return;
    }
    const panel = $(".compare-panel");
    panel.scrollIntoView({ behavior: prefersReducedMotion.matches ? "auto" : "smooth", block: "start" });
    showToast(`已载入 ${state.selectedRuns.size} 条实验的对比上下文。`);
  });
}

function renderArchive() {
  const list = $("#recordList");
  if (!list) return;
  const filtered = archiveRecords.filter((record) => {
    const matchesStatus = state.archiveStatus === "all" || record.status === state.archiveStatus;
    const haystack = `${record.name} ${record.model} ${record.date} ${record.hardware} ${record.note}`.toLowerCase();
    const matchesSearch = !state.archiveSearch || haystack.includes(state.archiveSearch);
    return matchesStatus && matchesSearch;
  });

  $("#archiveCount").textContent = `${filtered.length} 条可见记录`;

  if (!filtered.length) {
    list.innerHTML = '<div class="empty-records">没有匹配的运行记录，尝试清除搜索或切换状态。</div>';
  } else {
    list.innerHTML = filtered
      .map((record) => {
        const selected = state.selectedRuns.has(record.id);
        const statusText = { completed: "已完成", running: "运行中", failed: "异常" }[record.status];
        return `
          <article class="record-row${selected ? " is-selected" : ""}" data-record-id="${record.id}">
            <input
              class="record-check"
              type="checkbox"
              ${selected ? "checked" : ""}
              aria-label="将 ${escapeHtml(record.name)} 加入对比"
              data-compare-id="${record.id}"
            />
            <div class="record-main">
              <div class="record-title-row">
                <strong>${escapeHtml(record.name)}</strong>
                <span class="status-badge status-${record.status}">${statusText}</span>
              </div>
              <span class="record-model">${escapeHtml(record.model)} · ${record.runs} 次运行</span>
            </div>
            <div class="record-meta">
              <span>${escapeHtml(record.date)}</span>
              <span>${escapeHtml(record.hardware)} · ${record.duration}</span>
            </div>
            <div class="record-metric">
              <strong>${record.primaryValue.toFixed(2)}%</strong>
              <span>${escapeHtml(record.primaryLabel)}</span>
            </div>
            <button class="icon-button record-open" type="button" aria-label="载入 ${escapeHtml(record.name)} 上下文" data-open-record="${record.id}">
              <i data-lucide="arrow-up-right" aria-hidden="true"></i>
            </button>
          </article>
        `;
      })
      .join("");
  }

  $$(".record-check", list).forEach((input) => {
    input.addEventListener("change", () => {
      if (input.checked) {
        state.selectedRuns.add(input.dataset.compareId);
      } else {
        state.selectedRuns.delete(input.dataset.compareId);
      }
      renderArchive();
    });
  });

  $$(".record-open", list).forEach((button) => {
    button.addEventListener("click", () => openArchiveRecord(button.dataset.openRecord));
  });

  renderComparePanel();
  window.lucide?.createIcons({ attrs: { "stroke-width": 1.8 } });
}

function openArchiveRecord(recordId) {
  const record = archiveRecords.find((item) => item.id === recordId);
  if (!record) return;

  if (record.status === "running") {
    switchView("train");
  } else {
    switchView("inspect");
    if (record.weights?.seg && segmentationModels[record.weights.seg]) {
      state.segmentModel = segmentationModels[record.weights.seg];
      $$(".weight-option[data-seg]").forEach((button) => {
        const selected = button.dataset.seg === record.weights.seg;
        button.classList.toggle("is-selected", selected);
        button.setAttribute("aria-pressed", String(selected));
      });
    }
    if (record.weights?.rec && recognitionModels[record.weights.rec]) {
      state.recognitionModel = recognitionModels[record.weights.rec];
      $$(".weight-option[data-rec]").forEach((button) => {
        const selected = button.dataset.rec === record.weights.rec;
        button.classList.toggle("is-selected", selected);
        button.setAttribute("aria-pressed", String(selected));
      });
    }
    state.segProbability = null;
    state.segGray = null;
    state.mask = null;
    state.analysis = null;
    window.particlePipe?.setDefectsFromAnalysis(null);
    updateInferenceUI();
  }
  showToast(`已载入 ${record.name}：${record.note}`);
}

function renderComparePanel() {
  const selected = archiveRecords.filter((record) => state.selectedRuns.has(record.id));
  $("#compareCount").textContent = `${selected.length} 项`;
  $("#compareHint").textContent =
    selected.length < 2 ? "至少勾选两条记录以建立可比较上下文。" : "主指标按各自任务类型展示，请在表中核对指标口径。";
  $("#openCompareBtn").disabled = selected.length < 2;

  const chart = $("#compareChart");
  if (!selected.length) {
    chart.innerHTML = '<div class="empty-records">尚未选择实验。</div>';
  } else {
    chart.innerHTML = selected
      .map(
        (record) => `
          <div class="compare-bar">
            <span>${escapeHtml(record.name)}</span>
            <span class="bar-track" aria-hidden="true">
              <span style="width:${clamp(record.primaryValue, 0, 100).toFixed(2)}%"></span>
            </span>
            <strong>${record.primaryValue.toFixed(2)}%</strong>
          </div>
        `,
      )
      .join("");
  }

  $("#compareTableBody").innerHTML = selected.length
    ? selected
        .map(
          (record) => `
            <tr>
              <td>${escapeHtml(record.name)}</td>
              <td>${escapeHtml(record.primaryLabel)} ${record.primaryValue.toFixed(2)}%</td>
              <td>${escapeHtml(record.hardware)}</td>
              <td>${record.duration}</td>
            </tr>
          `,
        )
        .join("")
    : '<tr><td colspan="4">选择记录后显示对比表。</td></tr>';
}

function setupGlobalShortcuts() {
  document.addEventListener("keydown", (event) => {
    const target = event.target;
    const isTyping =
      target instanceof HTMLInputElement ||
      target instanceof HTMLTextAreaElement ||
      target instanceof HTMLSelectElement ||
      target?.isContentEditable;

    if (event.key === "/" && !isTyping) {
      event.preventDefault();
      switchView("archive");
      window.setTimeout(() => $("#archiveSearch").focus(), 80);
      return;
    }

    if (event.key === "Enter" && state.view === "inspect" && !isTyping && !state.inferenceRunning) {
      if (target instanceof HTMLButtonElement) return;
      event.preventDefault();
      runInference();
      return;
    }

    if (isTyping || event.ctrlKey || event.metaKey || event.altKey) return;
    if (event.key === "1") switchView("inspect");
    if (event.key === "2") switchView("train");
    if (event.key === "3") switchView("archive");
    if (event.key === "Escape") closeReport();
  });

  let resizeTimer = null;
  window.addEventListener("resize", () => {
    window.clearTimeout(resizeTimer);
    resizeTimer = window.setTimeout(() => {
      if (!window.matchMedia("(max-width: 900px)").matches && !$("#reportDrawer").classList.contains("is-open")) {
        openReport();
      }
    }, 180);
  });
}

function renderCode(code, filename) {
  const normalized = code.replace(/\r\n/g, "\n");
  const lines = normalized.split("\n");
  $("#codeFilename").textContent = filename;
  $("#codePreview").innerHTML = highlightCode(normalized);
  $("#codeLineNumbers").textContent = lines.map((_, index) => index + 1).join("\n");
}

function highlightCode(code) {
  const preserved = [];
  const reserve = (html) => {
    const token = `\u0000TOKEN${preserved.length}\u0000`;
    preserved.push(html);
    return token;
  };

  let escaped = escapeHtml(code);
  escaped = escaped.replace(/(^|\n)(\s*)(#.*)/g, (_, prefix, spaces, comment) => {
    return `${prefix}${spaces}${reserve(`<span class="code-comment">${comment}</span>`)}`;
  });
  escaped = escaped.replace(/(&quot;.*?&quot;|&#39;.*?&#39;)/g, (match) => {
    return reserve(`<span class="code-string">${match}</span>`);
  });
  escaped = escaped.replace(
    /\b(from|import|as|def|class|return|if|else|for|while|with|in|is|None|True|False|and|or|not|try|except|print)\b/g,
    '<span class="code-keyword">$1</span>',
  );

  preserved.forEach((html, index) => {
    escaped = escaped.replaceAll(`\u0000TOKEN${index}\u0000`, html);
  });
  return escaped;
}

function showToast(message, type = "default") {
  const region = $("#toastRegion");
  while (region.firstElementChild) {
    region.firstElementChild.remove();
  }
  const toast = document.createElement("div");
  toast.className = `toast${type === "warning" ? " is-warning" : ""}`;
  toast.innerHTML = `
    <i data-lucide="${type === "warning" ? "circle-alert" : "sparkles"}" aria-hidden="true"></i>
    <span>${escapeHtml(message)}</span>
  `;
  region.appendChild(toast);
  window.lucide?.createIcons({ attrs: { "stroke-width": 1.8 } });

  window.setTimeout(() => {
    toast.style.opacity = "0";
    toast.style.transform = "translateY(6px)";
    window.setTimeout(() => toast.remove(), prefersReducedMotion.matches ? 0 : 180);
  }, 3800);
}

function formatBytes(bytes) {
  if (!bytes) return "0 B";
  const units = ["B", "KB", "MB", "GB"];
  const index = Math.min(units.length - 1, Math.floor(Math.log(bytes) / Math.log(1024)));
  return `${(bytes / Math.pow(1024, index)).toFixed(index > 1 ? 1 : 0)} ${units[index]}`;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

init();
