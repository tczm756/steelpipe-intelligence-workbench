#!/usr/bin/env node

import { createHash } from "node:crypto";
import { execFileSync } from "node:child_process";
import {
  existsSync,
  readFileSync,
  readdirSync,
  statSync,
  writeFileSync,
} from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const SCRIPT_DIR = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(SCRIPT_DIR, "..");
const MODEL_ROOT = path.join(ROOT, "models");
const MANIFEST_PATH = path.join(MODEL_ROOT, "manifest.json");
const LOCK_PATH = path.join(ROOT, "public-weights.lock.json");
const UPDATE = process.argv.includes("--update");
const CHECK_HISTORY = process.argv.includes("--history");

const PUBLIC_WEIGHTS = new Map([
  ["segmentation-boundary.onnx", { id: "small-unet-boundary", kind: "segmentation" }],
  ["segmentation-original.onnx", { id: "small-unet-original", kind: "segmentation" }],
  ["segmentation-ablation.onnx", { id: "small-unet-ablation", kind: "segmentation" }],
  ["recognition-mobilenetv2.onnx", { id: "mobilenetv2", kind: "recognition" }],
  ["recognition-small-ablation.onnx", { id: "small-classifier", kind: "recognition" }],
  ["recognition-small-original.onnx", { id: "small-classifier-original", kind: "recognition" }],
]);

const ALLOWED_WEIGHT_FILES = new Set(
  [...PUBLIC_WEIGHTS.keys()].flatMap((name) => [name, `${name}.gz`]),
);

const FORBIDDEN_EXTENSIONS = new Set([
  ".pt",
  ".pth",
  ".ckpt",
  ".safetensors",
  ".pkl",
]);

const PRIVATE_NAME_PATTERN = /(?:^|[._-])(latest|private|secret|confidential)(?:[._-]|$)/i;

function fail(message) {
  console.error(`PUBLIC WEIGHT GUARD: ${message}`);
  process.exit(1);
}

function sha256(filePath) {
  return createHash("sha256").update(readFileSync(filePath)).digest("hex");
}

function walk(directory) {
  const files = [];
  for (const entry of readdirSync(directory)) {
    const fullPath = path.join(directory, entry);
    const stat = statSync(fullPath);
    if (stat.isDirectory()) {
      files.push(...walk(fullPath));
    } else {
      files.push(fullPath);
    }
  }
  return files;
}

function stripInternalProvenance(value) {
  if (Array.isArray(value)) {
    return value.map(stripInternalProvenance);
  }
  if (value && typeof value === "object") {
    return Object.fromEntries(
      Object.entries(value)
        .filter(([key]) => !/sourceCheckpoint/i.test(key))
        .map(([key, child]) => [key, stripInternalProvenance(child)]),
    );
  }
  return value;
}

function assertPublicManifest(manifest) {
  if (manifest.weightVisibility !== "public-benchmark") {
    fail("models/manifest.json must declare weightVisibility=public-benchmark.");
  }

  const entries = [
    ...(manifest.segmentation || []).map((entry) => ({ ...entry, kind: "segmentation" })),
    ...(manifest.recognition || []).map((entry) => ({ ...entry, kind: "recognition" })),
  ];

  if (entries.length !== PUBLIC_WEIGHTS.size) {
    fail(`Expected ${PUBLIC_WEIGHTS.size} public weights, found ${entries.length}.`);
  }

  const manifestFiles = new Set();
  for (const entry of entries) {
    const fileName = path.basename(entry.file || "");
    const expected = PUBLIC_WEIGHTS.get(fileName);
    if (!expected) {
      fail(`Unexpected public weight in manifest: ${entry.file || "<missing>"}.`);
    }
    if (entry.id !== expected.id || entry.kind !== expected.kind) {
      fail(`Manifest identity mismatch for ${fileName}.`);
    }
    manifestFiles.add(fileName);
  }

  for (const fileName of PUBLIC_WEIGHTS.keys()) {
    if (!manifestFiles.has(fileName)) {
      fail(`Approved public weight missing from manifest: ${fileName}.`);
    }
  }

  const serialized = JSON.stringify(manifest);
  if (/sourceCheckpoint/i.test(serialized)) {
    fail("models/manifest.json contains internal checkpoint provenance.");
  }
}

function assertCurrentModelDirectory() {
  const files = walk(MODEL_ROOT);
  const presentWeights = new Set();

  for (const filePath of files) {
    const fileName = path.basename(filePath);
    const relative = path.relative(ROOT, filePath).replaceAll("\\", "/");
    const extension = path.extname(fileName).toLowerCase();

    if (relative === "models/manifest.json") continue;
    if (FORBIDDEN_EXTENSIONS.has(extension)) {
      fail(`Forbidden private checkpoint format: ${relative}.`);
    }
    if (PRIVATE_NAME_PATTERN.test(fileName)) {
      fail(`Private-looking weight name: ${relative}.`);
    }
    if (extension === ".onnx" || fileName.endsWith(".onnx.gz")) {
      if (!ALLOWED_WEIGHT_FILES.has(fileName)) {
        fail(`Weight is not in the public allowlist: ${relative}.`);
      }
      presentWeights.add(fileName);
    }
  }

  for (const fileName of ALLOWED_WEIGHT_FILES) {
    if (!presentWeights.has(fileName)) {
      fail(`Expected public weight file is missing: models/${fileName}.`);
    }
  }
}

function assertHistory() {
  if (!CHECK_HISTORY) return;
  let output;
  try {
    output = execFileSync(
      "git",
      ["log", "--all", "--pretty=format:", "--name-only"],
      { cwd: ROOT, encoding: "utf8" },
    );
  } catch (error) {
    fail(`Unable to inspect public repository history: ${error.message}`);
  }

  const historicalPaths = new Set(
    output.split(/\r?\n/).map((value) => value.trim()).filter(Boolean),
  );

  for (const relative of historicalPaths) {
    if (!relative.startsWith("models/")) continue;
    const fileName = path.basename(relative);
    const extension = path.extname(fileName).toLowerCase();
    if (FORBIDDEN_EXTENSIONS.has(extension)) {
      fail(`Forbidden checkpoint exists in Git history: ${relative}.`);
    }
    if (PRIVATE_NAME_PATTERN.test(fileName)) {
      fail(`Private-looking weight exists in Git history: ${relative}.`);
    }
    if (
      (extension === ".onnx" || fileName.endsWith(".onnx.gz")) &&
      !ALLOWED_WEIGHT_FILES.has(fileName)
    ) {
      fail(`Non-allowlisted weight exists in Git history: ${relative}.`);
    }
  }
}

function buildLock(manifest) {
  const models = [
    ...manifest.segmentation.map((entry) => ({ ...entry, kind: "segmentation" })),
    ...manifest.recognition.map((entry) => ({ ...entry, kind: "recognition" })),
  ]
    .map((entry) => {
      const fileName = path.basename(entry.file);
      const modelPath = path.join(MODEL_ROOT, fileName);
      const gzipPath = `${modelPath}.gz`;
      return {
        id: entry.id,
        kind: entry.kind,
        displayName: entry.displayName,
        file: `models/${fileName}`,
        sha256: sha256(modelPath),
        compressedFile: `models/${fileName}.gz`,
        compressedSha256: sha256(gzipPath),
      };
    })
    .sort((left, right) => left.file.localeCompare(right.file));

  return {
    formatVersion: 1,
    weightVisibility: "public-benchmark",
    publicSet: "SmallUNet and published comparison checkpoints",
    forbiddenExtensions: [...FORBIDDEN_EXTENSIONS].sort(),
    models,
  };
}

if (!existsSync(MANIFEST_PATH)) {
  fail("models/manifest.json is missing.");
}

let manifest = JSON.parse(readFileSync(MANIFEST_PATH, "utf8"));
manifest = stripInternalProvenance(manifest);

if (UPDATE) {
  manifest.weightVisibility = "public-benchmark";
  writeFileSync(MANIFEST_PATH, `${JSON.stringify(manifest, null, 2)}\n`, "utf8");
}

assertPublicManifest(manifest);
assertCurrentModelDirectory();
assertHistory();

const lock = buildLock(manifest);
if (UPDATE) {
  writeFileSync(LOCK_PATH, `${JSON.stringify(lock, null, 2)}\n`, "utf8");
  console.log(`PUBLIC WEIGHT GUARD: updated ${LOCK_PATH}`);
} else if (!existsSync(LOCK_PATH)) {
  fail("public-weights.lock.json is missing.");
} else {
  const current = JSON.parse(readFileSync(LOCK_PATH, "utf8"));
  if (JSON.stringify(current) !== JSON.stringify(lock)) {
    fail("public-weights.lock.json is stale. Re-run with --update.");
  }
  console.log("PUBLIC WEIGHT GUARD: current files and repository history are clean.");
}
