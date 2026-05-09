from __future__ import annotations

import argparse
import json
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from flash_liveness_infer_utils import (
    detect_media_type,
    load_flash_liveness_bundle,
    parse_flash_colors,
    predict_from_tensor,
    tensor_from_media_path,
    tensor_from_uploaded_media,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Flash liveness API for video/image inference.")
    parser.add_argument(
        "--checkpoint",
        default="/supercloud/llm-code/scc/scc/Liveness_Detection/flash_liveness_runs/flash_liveness_gpu1_localresnet_e20_manual/best_flash_liveness_model.pth",
        help="best/last_flash_liveness_model.pth",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=18220)
    parser.add_argument("--device", default=None, help="例如 cuda:0/cpu")
    parser.add_argument("--threshold", type=float, default=None, help="默认读取 checkpoint 中保存的阈值")
    parser.add_argument("--detector-model", default=None, help="YOLOv7 人脸检测权重路径；为空则中心裁剪")
    parser.add_argument("--detector-device", default=None)
    parser.add_argument("--detector-conf", type=float, default=0.5)
    parser.add_argument("--detector-iou", type=float, default=0.5)
    parser.add_argument(
        "--static-sequence-mode",
        choices=["copy", "flash"],
        default="copy",
        help="图片输入转视频序列的方式：copy 为复制静态图，flash 为合成闪光序列。",
    )
    parser.add_argument("--flash-group-size", type=int, default=4)
    parser.add_argument("--flash-colors", default="normal,red,green,blue")
    parser.add_argument("--flash-alpha", type=float, default=0.25)
    return parser.parse_args()


def create_app(args: argparse.Namespace) -> FastAPI:
    bundle = load_flash_liveness_bundle(
        checkpoint_path=args.checkpoint,
        device_spec=args.device,
        detector_model=args.detector_model,
        detector_device=args.detector_device,
        detector_conf=args.detector_conf,
        detector_iou=args.detector_iou,
        threshold_override=args.threshold,
        corrupted_record_path=None,
    )
    flash_colors = parse_flash_colors(args.flash_colors)

    app = FastAPI(title="Flash Liveness API", version="1.0.0")

    @app.get("/health")
    async def health() -> dict:
        return {
            "status": "ok",
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "device": str(bundle["device"]),
            "threshold": bundle["threshold"],
            "num_frames": bundle["num_frames"],
            "target_size": list(bundle["target_size"]),
            "static_sequence_mode": args.static_sequence_mode,
        }

    @app.post("/predict")
    async def predict(
        file: UploadFile | None = File(default=None),
        path: str | None = Form(default=None),
    ) -> dict:
        if file is None and not path:
            raise HTTPException(status_code=400, detail="请提供上传文件 file 或本地路径 path。")
        if file is not None and path:
            raise HTTPException(status_code=400, detail="file 和 path 二选一即可。")

        try:
            if path:
                media_path = Path(path).expanduser().resolve()
                if not media_path.exists():
                    raise HTTPException(status_code=404, detail=f"文件不存在: {media_path}")
                tensor_frames, media_type = tensor_from_media_path(
                    media_path=media_path,
                    bundle=bundle,
                    static_sequence_mode=args.static_sequence_mode,
                    flash_group_size=args.flash_group_size,
                    flash_colors=flash_colors,
                    flash_alpha=args.flash_alpha,
                )
                source_name = media_path.name
                source_path = str(media_path)
            else:
                payload = await file.read()
                if not payload:
                    raise HTTPException(status_code=400, detail="上传文件为空。")
                source_name = file.filename or "upload.bin"
                media_type = detect_media_type(Path(source_name))
                tensor_frames, media_type = tensor_from_uploaded_media(
                    payload=payload,
                    filename_hint=source_name,
                    bundle=bundle,
                    static_sequence_mode=args.static_sequence_mode,
                    flash_group_size=args.flash_group_size,
                    flash_colors=flash_colors,
                    flash_alpha=args.flash_alpha,
                )
                source_path = ""

            prediction = predict_from_tensor(
                model=bundle["model"],
                tensor_frames=tensor_frames,
                device=bundle["device"],
                threshold=bundle["threshold"],
            )
            return {
                "result": prediction["prediction_name"],
                "prediction_id": prediction["prediction_id"],
                "probability_live": round(float(prediction["probability_live_raw"]), 8),
                "threshold": round(float(prediction["threshold"]), 8),
                "media_type": media_type,
                "source_name": source_name,
                "source_path": source_path,
                "num_frames_used": int(tensor_frames.shape[0]),
                "basis": [
                    f"checkpoint={Path(args.checkpoint).name}",
                    f"media_type={media_type}",
                    f"probability_live={float(prediction['probability_live_raw']):.8f}",
                    f"threshold={float(prediction['threshold']):.8f}",
                    f"static_sequence_mode={args.static_sequence_mode if media_type == 'image' else 'video_native'}",
                ],
            }
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.get("/")
    async def index() -> dict:
        return {
            "service": "flash_liveness_api",
            "endpoints": {
                "health": "/health",
                "predict": "/predict",
            },
            "usage": {
                "multipart_file": "POST /predict with form-data key=file",
                "local_path": "POST /predict with form-data key=path",
            },
        }

    return app


def main() -> None:
    args = parse_args()
    app = create_app(args)
    print(
        json.dumps(
            {
                "host": args.host,
                "port": args.port,
                "checkpoint": str(Path(args.checkpoint).resolve()),
                "device": args.device or "auto",
                "static_sequence_mode": args.static_sequence_mode,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
