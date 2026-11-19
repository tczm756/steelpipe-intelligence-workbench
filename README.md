# YOPO (You only peek once)

YOPO 是线激光钢管缺陷检测可视化工作台。推理页的公开权重集仅包含公开对比基准权重：SmallUNet 系列，以及 MobileNetV2、SmallDefectClassifier 等公开对比检查点。权重通过 ONNX Runtime Web 在浏览器本地执行。

## 启动

```powershell
.\serve.ps1
```

浏览器打开 `http://127.0.0.1:4173/`。

如果训练服务提供 SSE，可通过 `http://127.0.0.1:4173/?stream=http://127.0.0.1:9000/training/events` 接入。默认事件 `snapshot` 或 `message` 接受 `progress`、`epoch`、`loss`、`dice`、`status` 字段；`log` 事件接受 `message`、`level` 字段。

连接格式、GPU 遥测、数据集摘要和控制端点见 `CLOUD_TRAINING.md`。

## 工作层

- `推理勘探`：上传图像后，用所选分割权重生成 320×320 概率图；阈值化后按掩码边界外扩 20%，再用所选识别权重生成六类 softmax 概率。粒子缺陷由真实掩码像素映射生成，初始页面不执行推理。
- `云端训练`：实时训练进度、阶段状态、损失/Dice 曲线、数据集上传、代码预览和超参数调节。
- `实验档案`：搜索历史运行、按状态筛选、勾选记录并比较主指标与上下文。

当前公开可选的 6 个推理权重：

```text
segmentation-boundary.onnx
segmentation-original.onnx
segmentation-ablation.onnx
recognition-mobilenetv2.onnx
recognition-small-ablation.onnx
recognition-small-original.onnx
```

公开权重清单、SHA256、训练指标和预处理定义记录在 `models/manifest.json` 与 `public-weights.lock.json`。浏览器输入与 PyTorch 一致：灰度转换、`(pixel / 255 - 0.5) / 0.5`、分割 320×320、识别 128×128 双通道。浏览器重采样与 Pillow 的重采样实现略有差异，阈值边界附近可能相差少量像素；同一张 `In_1.bmp` 在阈值 `0.42` 下，PyTorch 参考掩码为 2,032 像素，线上浏览器实测为 2,051 像素，两者均识别为 `inclusion`。

重新导出权重：

```powershell
cd "01_程序与工程"
$env:PYTHONPATH = "src"
.\.onnx-venv\Scripts\python.exe scripts\export_workbench_onnx.py
```

未连接真实状态流时，云端训练页显示 `NO PROJECT`，不会生成训练进度、GPU 指标、事件或检查点。数据集文件可先在浏览器中登记，但不会伪装成已上传。推理页的图像上传和 ONNX 计算使用真实权重。
