# Cloud Training Connection Contract

The workbench does not create fake runs. Without a real SSE endpoint, the training page shows `NO PROJECT` and keeps telemetry, GPU metrics, event logs, and archive mutations empty.

## Connect

Use the connection form in the training view or open:

```text
http://127.0.0.1:4173/?stream=http://127.0.0.1:9000/training/events
```

Never put an SSH password or private key in the browser. The training host should expose a local or authenticated SSE endpoint through a broker.

## Snapshot event

Event name: `snapshot`

```json
{
  "project": {
    "run_id": "robust-mask-v1-seed42",
    "model": "SmallDefectRobustClassifier",
    "seed": 42,
    "hardware": "RTX 4090D",
    "eta": "预计 18 分钟",
    "controls": {
      "pause": "http://127.0.0.1:9000/training/pause",
      "resume": "http://127.0.0.1:9000/training/resume",
      "start": "http://127.0.0.1:9000/training/start"
    },
    "dataset": {
      "name": "robust-mask-v1",
      "pipes": 12,
      "sequences": 36,
      "split": "3679 / 720 / 690",
      "leakCheck": "无泄漏",
      "state": "按钢管编号分组"
    }
  },
  "status": "running",
  "progress": 34.2,
  "epoch": 22,
  "loss": 0.093,
  "dice": 0.871,
  "vitals": {
    "gpuLoad": 91,
    "vramUsedGb": 18.4,
    "vramTotalGb": 24,
    "throughput": 126,
    "temperature": 68,
    "dataloader": 84,
    "bandwidth": 63,
    "checkpoint": "epoch_022.pt",
    "checkpointMeta": "已写入 · 9.1 MB"
  }
}
```

The same numeric fields may also be sent at the top level. Control endpoints are optional. Without them, the UI remains observation-only.

## Log event

Event name: `log`

```json
{
  "level": "ok",
  "message": "validation complete · mask robustness macro-F1 0.982"
}
```

Use `level: "warn"` for recoverable failures or contract violations.

## Failure behavior

- SSE disconnect: clear the project and telemetry, show `NO PROJECT`, and ask the user to reconnect.
- Invalid JSON snapshot: show an error status; keep any previously received real project.
- No control endpoint: disable start/pause controls.
- No stream endpoint: never synthesize training progress or GPU metrics.
