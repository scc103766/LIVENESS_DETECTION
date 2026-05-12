from __future__ import annotations

import argparse
import html
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from flash_liveness_v3_api_service.service import (  # noqa: E402
    DEFAULT_CHECKPOINT_PATH,
    DEFAULT_OUTPUT_DIR,
    FUSION_METHODS,
    FlashLivenessV3ApiService,
    infer_upload_type,
)


class PredictPathRequest(BaseModel):
    video_path: str
    txt_path: Optional[str] = None
    request_id: Optional[str] = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Flash Liveness V3 fixed-protocol API service.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=18131)
    parser.add_argument("--checkpoint-path", default=str(DEFAULT_CHECKPOINT_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--storage-backup-dir",
        default=None,
        help="Directory for retention backups. Default: <output-dir>/_retention_backups.",
    )
    parser.add_argument("--storage-max-videos", type=int, default=2000)
    parser.add_argument("--storage-cleanup-batch-size", type=int, default=200)
    parser.add_argument("--disable-storage-retention", action="store_true")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--device", default=None, help="Default: cuda if available, else cpu.")
    parser.add_argument("--inference-mode", choices=["window", "full_sequence"], default="window")
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--window-size", type=int, default=256)
    parser.add_argument("--window-stride", type=int, default=128)
    parser.add_argument("--window-fusion", choices=sorted(FUSION_METHODS), default="quality_lower_trimmed_mean")
    parser.add_argument("--window-trim-ratio", type=float, default=0.2)
    parser.add_argument("--window-min-quality", type=float, default=0.05)
    parser.add_argument("--require-color-txt", action="store_true")
    parser.add_argument("--detector-model", default=None)
    parser.add_argument("--detector-device", default=None)
    parser.add_argument("--detector-conf", type=float, default=0.5)
    parser.add_argument("--detector-iou", type=float, default=0.5)
    parser.add_argument("--show-decoder-warnings", action="store_true")
    parser.add_argument("--return-window-details", action="store_true")
    return parser.parse_args()


def render_home_page() -> str:
    return """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Flash Liveness V3 API</title>
  <style>
    body { margin: 0; font-family: sans-serif; background: #f6f7f8; color: #16181d; }
    main { max-width: 980px; margin: 0 auto; padding: 32px 20px; }
    section { background: #fff; border: 1px solid #d9dee5; border-radius: 8px; padding: 24px; }
    h1 { margin: 0 0 10px; font-size: 28px; }
    p { color: #4f5a67; line-height: 1.7; }
    form { display: grid; gap: 12px; margin: 18px 0; }
    input[type=file] { padding: 10px; border: 1px dashed #b9c1cc; border-radius: 8px; background: #fbfcfd; }
    button { width: fit-content; border: 0; border-radius: 8px; padding: 11px 18px; background: #175c4c; color: #fff; cursor: pointer; }
    pre { white-space: pre-wrap; word-break: break-word; background: #14181f; color: #f4f7fb; border-radius: 8px; padding: 16px; min-height: 260px; }
    label { font-weight: 700; }
  </style>
</head>
<body>
  <main>
    <section>
      <h1>Flash Liveness V3 固定协议活体检测 API</h1>
      <p>上传视频，或上传包含同名视频和 txt 的 zip。txt 缺失时服务会按 V3 训练配置自动生成 collect_flash 固定协议颜色时间线。</p>
      <form id="predict-form">
        <label>视频或 zip</label>
        <input type="file" id="file" name="file" accept="video/*,.zip" required>
        <label>可选同名 txt</label>
        <input type="file" id="txt-file" name="txt_file" accept=".txt">
        <button type="submit">调用 /predict</button>
      </form>
      <pre id="result-box">等待上传文件...</pre>
    </section>
  </main>
  <script>
    const form = document.getElementById("predict-form");
    const resultBox = document.getElementById("result-box");
    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      const fileInput = document.getElementById("file");
      const txtInput = document.getElementById("txt-file");
      const formData = new FormData();
      formData.append("file", fileInput.files[0]);
      if (txtInput.files.length) {
        formData.append("txt_file", txtInput.files[0]);
      }
      resultBox.textContent = "正在处理，请稍候...";
      try {
        const response = await fetch("/predict", { method: "POST", body: formData });
        const payload = await response.json();
        resultBox.textContent = JSON.stringify(payload, null, 2);
      } catch (error) {
        resultBox.textContent = "请求失败: " + error;
      }
    });
  </script>
</body>
</html>"""


def create_app(args: argparse.Namespace) -> FastAPI:
    service = FlashLivenessV3ApiService(
        checkpoint_path=args.checkpoint_path,
        output_dir=args.output_dir,
        threshold=args.threshold,
        device=args.device,
        inference_mode=args.inference_mode,
        max_seq_len=args.max_seq_len,
        window_size=args.window_size,
        window_stride=args.window_stride,
        window_fusion=args.window_fusion,
        window_trim_ratio=args.window_trim_ratio,
        window_min_quality=args.window_min_quality,
        require_color_txt=args.require_color_txt,
        detector_model_path=args.detector_model,
        detector_device=args.detector_device,
        detector_conf=args.detector_conf,
        detector_iou=args.detector_iou,
        show_decoder_warnings=args.show_decoder_warnings,
        return_window_details=args.return_window_details,
        storage_max_videos=args.storage_max_videos,
        storage_backup_dir=args.storage_backup_dir,
        storage_cleanup_batch_size=args.storage_cleanup_batch_size,
        storage_retention_enabled=not args.disable_storage_retention,
    )
    app = FastAPI(title="Flash Liveness V3 API Service")

    @app.get("/health")
    async def health() -> dict:
        return {
            "status": "ok",
            "service": "flash_liveness_v3_api_service",
            "version": "v3_fixed_protocol",
            "checkpoint_path": str(service.checkpoint_path),
            "threshold": service.threshold,
            "device": str(service.device),
            "inference_mode": service.inference_mode,
            "window_size": service.window_size,
            "window_stride": service.window_stride,
            "window_fusion": service.window_fusion,
            "require_color_txt": service.require_color_txt,
            "flash_protocol": {
                "missing_color_protocol": service.config.get("missing_color_protocol", "collect_flash"),
                "warmup_seconds": service.config.get("flash_warmup_seconds", 1.0),
                "hold_seconds": service.config.get("flash_hold_seconds", 0.35),
                "restore_seconds": service.config.get("flash_restore_seconds", 0.0),
                "tail_seconds": service.config.get("flash_tail_seconds", 0.5),
                "color_order_packed": [16717055, 1376020, 16716820],
            },
            "output_dir": str(Path(args.output_dir).resolve()),
            "storage_backup_dir": str(Path(args.storage_backup_dir).resolve())
            if args.storage_backup_dir
            else str((Path(args.output_dir).resolve() / "_retention_backups")),
            "storage_max_videos": args.storage_max_videos,
            "storage_retention_enabled": not args.disable_storage_retention,
        }

    @app.get("/", response_class=HTMLResponse)
    async def home() -> HTMLResponse:
        return HTMLResponse(render_home_page())

    @app.post("/predict")
    async def predict(file: UploadFile = File(...), txt_file: Optional[UploadFile] = File(default=None)) -> JSONResponse:
        suffix = Path(file.filename or "upload.bin").suffix.lower()
        try:
            input_type = infer_upload_type(file.filename or "")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        if input_type == "archive" and txt_file is not None:
            raise HTTPException(status_code=400, detail="txt_file is only supported when file is a video")

        with tempfile.TemporaryDirectory(prefix="flash_v3_api_") as temp_dir:
            temp_path = Path(temp_dir) / f"upload{suffix}"
            with temp_path.open("wb") as output:
                shutil.copyfileobj(file.file, output)

            temp_txt_path = None
            if txt_file is not None:
                if Path(txt_file.filename or "upload.txt").suffix.lower() != ".txt":
                    raise HTTPException(status_code=400, detail="txt_file must be a .txt file")
                temp_txt_path = Path(temp_dir) / "upload.txt"
                with temp_txt_path.open("wb") as output:
                    shutil.copyfileobj(txt_file.file, output)

            try:
                if input_type == "archive":
                    payload = service.predict_archive(temp_path, uploaded_filename=html.escape(file.filename or ""))
                else:
                    payload = service.predict_video(
                        temp_path,
                        txt_path=temp_txt_path,
                        uploaded_filename=html.escape(file.filename or ""),
                    )
            except Exception as exc:
                raise HTTPException(status_code=500, detail=f"inference_failed:{exc}") from exc

        payload["api_version"] = "v3_fixed_protocol"
        return JSONResponse(payload)

    @app.post("/predict_path")
    async def predict_path(request: PredictPathRequest) -> JSONResponse:
        try:
            payload = service.predict_path(
                Path(request.video_path),
                txt_path=Path(request.txt_path) if request.txt_path else None,
                request_id=request.request_id,
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"inference_failed:{exc}") from exc
        payload["api_version"] = "v3_fixed_protocol"
        return JSONResponse(payload)

    return app


def main() -> None:
    args = parse_args()
    app = create_app(args)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
