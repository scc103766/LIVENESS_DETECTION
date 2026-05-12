from __future__ import annotations

import argparse
import json
import shutil
import socket
import sys
import time
import uuid
import zipfile
from pathlib import Path
from typing import Any

import cv2
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.collect_flash_liveness_video import (  # noqa: E402
    COLOR_SEQUENCE_RGB,
    build_frame_color_labels,
    rgb_to_packed_int,
)


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "flash_collect_stimulus_api_service" / "outputs"
RECOMMENDED_TOTAL_SECONDS = 3.0
TRAINING_COLOR_INDICES = [1, 2, 3]
TRAINING_COLOR_SEQUENCE_RGB = COLOR_SEQUENCE_RGB
TRAINING_WARMUP_SECONDS = 1.0
TRAINING_HOLD_SECONDS = 0.35
TRAINING_RESTORE_SECONDS = 0.0
TRAINING_TAIL_SECONDS = 0.5
DEFAULT_FALLBACK_FPS = 30.0
FLOAT_TOLERANCE = 1e-6
DEFAULT_PALETTE: dict[int, tuple[int, int, int]] = {
    index + 1: color for index, color in enumerate(COLOR_SEQUENCE_RGB)
}
VIDEO_EXTENSIONS = {".webm", ".mp4", ".mov", ".mkv", ".avi", ".m4v"}
SAFE_FILE_STEM_CHARS = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-")


class StimulusRequest(BaseModel):
    color_indices: list[int] | None = None
    colors_rgb: list[list[int]] | None = None
    total_seconds: float = RECOMMENDED_TOTAL_SECONDS
    warmup_seconds: float = TRAINING_WARMUP_SECONDS
    hold_seconds: float = TRAINING_HOLD_SECONDS
    restore_seconds: float = TRAINING_RESTORE_SECONDS
    cycles: int = 1
    tail_seconds: float = TRAINING_TAIL_SECONDS
    fps: float = 30.0
    width: int = 1080
    height: int = 1920
    codec: str = "mp4v"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Flash stimulus browser-camera collection API.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=18132)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--ssl-certfile", default=None, help="Optional HTTPS certificate file for uvicorn.")
    parser.add_argument("--ssl-keyfile", default=None, help="Optional HTTPS private key file for uvicorn.")
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def model_to_dict(model: BaseModel) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()  # type: ignore[attr-defined]
    return model.dict()


def local_ip_candidates() -> list[str]:
    candidates = {"127.0.0.1"}
    try:
        hostname = socket.gethostname()
        for result in socket.getaddrinfo(hostname, None, socket.AF_INET):
            candidates.add(result[4][0])
    except OSError:
        pass
    return sorted(candidates)


def validate_rgb(raw: list[int]) -> tuple[int, int, int]:
    if len(raw) != 3:
        raise ValueError(f"RGB color must have exactly 3 values: {raw}")
    rgb = tuple(int(value) for value in raw)
    if any(value < 0 or value > 255 for value in rgb):
        raise ValueError(f"RGB values must be in [0, 255]: {raw}")
    return rgb  # type: ignore[return-value]


def resolve_color_sequence(request: StimulusRequest) -> list[tuple[int, int, int]]:
    if request.colors_rgb:
        colors = [validate_rgb(color) for color in request.colors_rgb]
        if colors != TRAINING_COLOR_SEQUENCE_RGB:
            raise ValueError(
                "colors_rgb must match the V3 training protocol: "
                f"{TRAINING_COLOR_SEQUENCE_RGB}"
            )
    else:
        indices = [int(index) for index in (request.color_indices or TRAINING_COLOR_INDICES)]
        if indices != TRAINING_COLOR_INDICES:
            raise ValueError(f"color_indices must be {TRAINING_COLOR_INDICES} for the V3 training protocol.")
        colors = [DEFAULT_PALETTE[index] for index in TRAINING_COLOR_INDICES]

    if not colors:
        raise ValueError("At least one flash color is required.")
    return colors


def require_float(value: float, expected: float, name: str) -> None:
    if abs(float(value) - float(expected)) > FLOAT_TOLERANCE:
        raise ValueError(f"{name} must be {expected} for the V3 training protocol.")


def validate_request(request: StimulusRequest) -> list[tuple[int, int, int]]:
    if request.warmup_seconds < 0 or request.tail_seconds < 0 or request.restore_seconds < 0:
        raise ValueError("warmup_seconds, restore_seconds and tail_seconds must be >= 0.")
    if request.hold_seconds <= 0:
        raise ValueError("hold_seconds must be > 0.")
    if request.total_seconds <= 0:
        raise ValueError("total_seconds must be > 0.")
    if request.total_seconds <= request.warmup_seconds + request.tail_seconds:
        raise ValueError("total_seconds must be greater than warmup_seconds + tail_seconds.")
    if request.cycles != 1:
        raise ValueError("cycles must be 1; this service repeats colors by total_seconds, not by a fixed cycle count.")
    require_float(request.warmup_seconds, TRAINING_WARMUP_SECONDS, "warmup_seconds")
    require_float(request.hold_seconds, TRAINING_HOLD_SECONDS, "hold_seconds")
    require_float(request.restore_seconds, TRAINING_RESTORE_SECONDS, "restore_seconds")
    require_float(request.tail_seconds, TRAINING_TAIL_SECONDS, "tail_seconds")
    if request.fps < 1 or request.fps > 120:
        raise ValueError("fps must be in [1, 120].")
    if request.width < 64 or request.height < 64 or request.width > 3840 or request.height > 3840:
        raise ValueError("width and height must be in [64, 3840].")
    if len(request.codec) != 4:
        raise ValueError("codec must be a 4-character FourCC string.")
    return resolve_color_sequence(request)


def video_metadata(video_path: Path, fallback_fps: float = DEFAULT_FALLBACK_FPS) -> tuple[int, float, bool]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"failed_to_open_video:{video_path}")
    used_decode_count = False
    try:
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if frame_count <= 0:
            used_decode_count = True
            frame_count = 0
            while True:
                ok, _ = cap.read()
                if not ok:
                    break
                frame_count += 1
    finally:
        cap.release()
    if fps <= 1e-6:
        fps = fallback_fps
    if frame_count <= 0:
        raise RuntimeError(f"no_frame_count:{video_path}")
    return frame_count, fps, used_decode_count


def write_fixed_color_txt(
    txt_path: Path,
    frame_count: int,
    fps: float,
    request: StimulusRequest,
    colors: list[tuple[int, int, int]],
) -> None:
    labels = build_frame_color_labels(
        frame_count=frame_count,
        fps=fps,
        warmup_seconds=request.warmup_seconds,
        hold_seconds=request.hold_seconds,
        restore_seconds=request.restore_seconds,
        tail_seconds=request.tail_seconds,
        color_sequence=colors,
    )
    txt_path.write_text(
        "".join(f"{frame_idx},{color_value}\n" for frame_idx, color_value in enumerate(labels)),
        encoding="utf-8",
    )


def write_bundle_zip(bundle_path: Path, entries: list[tuple[Path, str]]) -> None:
    ensure_dir(bundle_path.parent)
    with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for source_path, archive_name in entries:
            if not source_path.exists():
                raise FileNotFoundError(str(source_path))
            archive.write(source_path, archive_name)


def request_from_metadata(metadata: dict[str, Any]) -> StimulusRequest:
    raw_request = metadata.get("request") or {}
    if not isinstance(raw_request, dict):
        raw_request = {}
    return StimulusRequest(**raw_request)


def packed_int_to_rgb(value: int) -> tuple[int, int, int]:
    return (value >> 16) & 0xFF, (value >> 8) & 0xFF, value & 0xFF


def build_stimulus_metadata(
    request: StimulusRequest,
    colors: list[tuple[int, int, int]],
) -> dict[str, Any]:
    total_frames = max(1, int(round(request.total_seconds * request.fps)))
    frame_labels = build_frame_color_labels(
        frame_count=total_frames,
        fps=request.fps,
        warmup_seconds=request.warmup_seconds,
        hold_seconds=request.hold_seconds,
        restore_seconds=request.restore_seconds,
        tail_seconds=request.tail_seconds,
        color_sequence=colors,
    )
    timeline: list[dict[str, Any]] = []
    run_start = 0
    flash_index = 1
    for frame_index, packed in enumerate(frame_labels):
        is_run_end = frame_index == len(frame_labels) - 1 or frame_labels[frame_index + 1] != packed
        if not is_run_end:
            continue

        rgb = (0, 0, 0) if packed == 0 else packed_int_to_rgb(packed)
        start_seconds = run_start / request.fps
        end_seconds = (frame_index + 1) / request.fps
        if packed == 0 and start_seconds < request.warmup_seconds:
            phase = "warmup"
        elif packed == 0 and end_seconds > request.total_seconds - request.tail_seconds:
            phase = "tail"
        elif packed == 0:
            phase = f"restore_{flash_index - 1}"
        else:
            phase = f"flash_{flash_index}"
            flash_index += 1
        timeline.append(
            {
                "phase": phase,
                "rgb": list(rgb),
                "color_int": int(packed),
                "duration_seconds": float(end_seconds - start_seconds),
                "requested_start_seconds": float(start_seconds),
                "requested_end_seconds": float(end_seconds),
                "start_frame": int(run_start),
                "end_frame": int(frame_index),
                "start_seconds": float(start_seconds),
                "end_seconds": float(end_seconds),
            }
        )
        run_start = frame_index + 1

    return {
        "frame_count": int(total_frames),
        "duration_seconds": float(total_frames / request.fps),
        "requested_duration_seconds": float(request.total_seconds),
        "fps": float(request.fps),
        "protocol_mode": "fixed_collect_protocol_timeline_only",
        "colors_rgb": [list(color) for color in colors],
        "color_ints": [rgb_to_packed_int(color) for color in colors],
        "timeline": timeline,
    }


def render_home_page() -> str:
    return """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
  <title>Flash Collect</title>
  <style>
    :root { color-scheme: light; --ink: #17201d; --muted: #5d6b66; --line: #d7dfdc; --brand: #0d6b57; --bg: #f4f7f6; }
    * { box-sizing: border-box; }
    body { margin: 0; min-height: 100vh; font-family: Arial, "PingFang SC", sans-serif; background: var(--bg); color: var(--ink); }
    header { padding: 20px 22px 12px; border-bottom: 1px solid var(--line); background: #fff; }
    main { max-width: 1180px; margin: 0 auto; padding: 18px; display: grid; gap: 18px; }
    h1 { margin: 0; font-size: 24px; font-weight: 760; }
    h2 { margin: 0 0 12px; font-size: 17px; }
    section { background: #fff; border: 1px solid var(--line); border-radius: 8px; padding: 16px; }
    .grid { display: grid; grid-template-columns: repeat(6, minmax(0, 1fr)); gap: 12px; }
    label { display: grid; gap: 6px; font-size: 13px; font-weight: 700; color: #23312d; }
    input, select { width: 100%; border: 1px solid #bfcac6; border-radius: 6px; padding: 9px 10px; font: inherit; }
    input[readonly] { background: #edf2f0; color: #43514d; }
    .hint { margin: 0 0 12px; color: var(--muted); font-size: 13px; line-height: 1.5; }
    button, a.button { border: 0; border-radius: 7px; padding: 10px 14px; background: var(--brand); color: #fff; font-weight: 700; cursor: pointer; text-decoration: none; display: inline-flex; align-items: center; justify-content: center; min-height: 38px; }
    button.secondary, a.secondary { background: #24332f; }
    button:disabled { opacity: .55; cursor: wait; }
    .actions { display: flex; gap: 10px; flex-wrap: wrap; margin-top: 14px; }
    .palette { display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 12px; }
    .swatch { display: inline-flex; align-items: center; gap: 8px; border: 1px solid var(--line); border-radius: 7px; padding: 9px 12px; background: #fff; color: #1e2d28; font-weight: 700; }
    .dot { width: 22px; height: 22px; border-radius: 50%; border: 1px solid rgba(0,0,0,.18); }
    pre { white-space: pre-wrap; word-break: break-word; background: #101715; color: #e8f3ee; border-radius: 8px; padding: 12px; min-height: 150px; font-size: 12px; }
    .flash-layer { position: fixed; inset: 0; z-index: 9999; background: #000; display: none; align-items: center; justify-content: center; transition: background-color 20ms linear; }
    .flash-layer.active { display: flex; }
    .capture-preview { --mirror-scale: 1; position: absolute; left: 50%; top: 50%; width: min(42vw, 320px); aspect-ratio: 9 / 16; object-fit: cover; display: none; border: 2px solid rgba(255,255,255,.72); border-radius: 8px; background: #000; box-shadow: 0 8px 26px rgba(0,0,0,.35); transform: translate(-50%, -50%) scaleX(var(--mirror-scale)); transform-origin: center; }
    .capture-preview.active { display: block; }
    .capture-preview.mirror { --mirror-scale: -1; }
    .flash-hud { position: fixed; left: 14px; right: 14px; bottom: 14px; z-index: 10001; display: none; min-height: 36px; padding: 9px 11px; border-radius: 6px; background: rgba(0,0,0,.42); color: #fff; font-size: 13px; line-height: 1.35; }
    .flash-hud.active { display: block; }
    @media (max-width: 860px) {
      .grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      header { padding: 16px; }
      main { padding: 12px; }
      .capture-preview { width: min(72vw, 260px); }
    }
  </style>
</head>
<body>
  <header><h1>Flash Collect</h1></header>
  <main>
    <section>
      <h2>刺激序列</h2>
      <p class="hint">协议固定为 fixed_collect_protocol，录制总时长可按业务输入；默认 3.0s 只是推荐值。</p>
      <div class="palette">
        <span class="swatch"><span class="dot" style="background: rgb(255,20,255)"></span>1</span>
        <span class="swatch"><span class="dot" style="background: rgb(20,255,20)"></span>2</span>
        <span class="swatch"><span class="dot" style="background: rgb(255,20,20)"></span>3</span>
      </div>
      <div class="grid">
        <label>颜色序号<input id="sequence" value="1,2,3" inputmode="numeric" readonly></label>
        <label>warmup 秒<input id="warmup" type="number" step="0.05" value="1.0" readonly></label>
        <label>hold 秒<input id="hold" type="number" step="0.05" value="0.35" readonly></label>
        <label>restore 秒<input id="restore" type="number" step="0.05" value="0.0" readonly></label>
        <label>tail 秒<input id="tail" type="number" step="0.05" value="0.5" readonly></label>
        <label>总时长秒<input id="total" type="number" step="0.05" value="3.0"></label>
        <label>fps<input id="fps" type="number" step="1" value="30"></label>
        <label>宽<input id="width" type="number" step="1" value="1080"></label>
        <label>高<input id="height" type="number" step="1" value="1920"></label>
        <label>摄像头方向<select id="facing-mode"><option value="user">前置</option><option value="environment">后置</option></select></label>
        <label>摄像头<select id="camera-select"><option value="">浏览器默认</option></select></label>
      </div>
      <div class="actions">
        <button id="create-btn" type="button">生成/刷新协议</button>
        <button id="record-btn" type="button" class="secondary">开始录制</button>
        <button id="refresh-camera-btn" type="button" class="secondary">刷新摄像头</button>
        <a id="download-recording-link" class="button secondary" href="#" download style="display:none">下载录制视频+TXT</a>
      </div>
    </section>
    <section>
      <h2>结果</h2>
      <pre id="result-box">ready</pre>
    </section>
  </main>
  <div id="flash-layer" class="flash-layer">
    <video id="capture-preview" class="capture-preview" autoplay muted playsinline></video>
  </div>
  <div id="flash-hud" class="flash-hud">ready</div>
  <script>
    const $ = (id) => document.getElementById(id);
    const resultBox = $("result-box");
    let session = null;
    let stream = null;

    function show(payload) {
      resultBox.textContent = typeof payload === "string" ? payload : JSON.stringify(payload, null, 2);
    }

    function sequence() {
      return $("sequence").value.split(",").map((item) => Number(item.trim())).filter((item) => Number.isFinite(item));
    }

    function sessionPayload() {
      return {
        color_indices: sequence(),
        total_seconds: $("total").value.trim() ? Number($("total").value) : 3.0,
        warmup_seconds: Number($("warmup").value),
        hold_seconds: Number($("hold").value),
        restore_seconds: Number($("restore").value),
        tail_seconds: Number($("tail").value),
        cycles: 1,
        fps: Number($("fps").value),
        width: Number($("width").value),
        height: Number($("height").value),
      };
    }

    async function createSession() {
      $("create-btn").disabled = true;
      try {
        const response = await fetch("/api/sessions", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify(sessionPayload()),
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || response.statusText);
        session = data;
        $("download-recording-link").style.display = "none";
        show(data);
        return data;
      } finally {
        $("create-btn").disabled = false;
      }
    }

    $("create-btn").addEventListener("click", async () => {
      try {
        await createSession();
      } catch (error) {
        $("create-btn").disabled = false;
        show(`生成失败: ${error}`);
      }
    });

    function pickMimeType() {
      const candidates = [
        "video/mp4;codecs=avc1.42E01E",
        "video/mp4",
        "video/webm;codecs=h264",
        "video/webm;codecs=vp9",
        "video/webm;codecs=vp8",
        "video/webm",
      ];
      return candidates.find((item) => window.MediaRecorder && MediaRecorder.isTypeSupported(item)) || "";
    }

    function sleep(ms) {
      return new Promise((resolve) => setTimeout(resolve, ms));
    }

    function rgbCss(rgb) {
      return `rgb(${rgb[0]}, ${rgb[1]}, ${rgb[2]})`;
    }

    function cameraUnavailableMessage() {
      const protocol = window.location.protocol;
      const host = window.location.hostname;
      const isLocalhost = host === "localhost" || host === "127.0.0.1" || host === "[::1]";
      const secure = window.isSecureContext || protocol === "https:" || isLocalhost;
      if (!secure) {
        return [
          "当前页面不是安全上下文，浏览器不会开放摄像头 API。",
          `手机访问服务器 IP 时不能使用普通 ${window.location.origin}。`,
          `请改用 HTTPS，或在 Android 调试时用 adb reverse 后访问 http://127.0.0.1:${window.location.port || "18132"}。`
        ].join("\\n");
      }
      return "当前浏览器没有提供 getUserMedia。请使用最新版 Chrome/Safari/Edge，并确认摄像头权限未被系统或浏览器禁用。";
    }

    function getUserMediaCompat(constraints) {
      if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
        return navigator.mediaDevices.getUserMedia(constraints);
      }
      const legacyGetUserMedia =
        navigator.getUserMedia ||
        navigator.webkitGetUserMedia ||
        navigator.mozGetUserMedia ||
        navigator.msGetUserMedia;
      if (legacyGetUserMedia) {
        return new Promise((resolve, reject) => legacyGetUserMedia.call(navigator, constraints, resolve, reject));
      }
      throw new Error(cameraUnavailableMessage());
    }

    async function attachPreview(cameraStream) {
      const preview = $("capture-preview");
      preview.srcObject = cameraStream;
      preview.classList.toggle("mirror", ($("facing-mode").value || "user") === "user");
      try {
        await preview.play();
      } catch (error) {
        /* preview may wait until the next user gesture on some browsers */
      }
    }

    function stopCamera() {
      const preview = $("capture-preview");
      preview.pause();
      preview.srcObject = null;
      preview.classList.remove("active", "mirror");
      if (stream) {
        stream.getTracks().forEach((track) => track.stop());
        stream = null;
      }
    }

    async function refreshCameras() {
      const select = $("camera-select");
      select.innerHTML = '<option value="">浏览器默认</option>';
      if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) {
        return;
      }
      const devices = await navigator.mediaDevices.enumerateDevices();
      devices.filter((device) => device.kind === "videoinput").forEach((device, index) => {
        const option = document.createElement("option");
        option.value = device.deviceId;
        option.textContent = device.label || `摄像头 ${index + 1}`;
        select.appendChild(option);
      });
    }

    async function ensureCamera() {
      if (stream) {
        await attachPreview(stream);
        return stream;
      }
      if (!window.MediaRecorder) {
        throw new Error("当前浏览器不支持 MediaRecorder，无法直接录制视频。请使用最新版 Chrome/Safari/Edge。");
      }
      const selectedDevice = $("camera-select").value;
      const facingMode = $("facing-mode").value || "user";
      const videoConstraints = selectedDevice
        ? {
            deviceId: { exact: selectedDevice },
            width: { ideal: 1080 },
            height: { ideal: 1920 },
            frameRate: { ideal: Number($("fps").value) || 30 },
          }
        : {
            facingMode,
            width: { ideal: 1080 },
            height: { ideal: 1920 },
            frameRate: { ideal: Number($("fps").value) || 30 },
          };
      stream = await getUserMediaCompat({
        video: videoConstraints,
        audio: false,
      });
      await attachPreview(stream);
      await refreshCameras();
      return stream;
    }

    function stopRecorder(recorder) {
      return new Promise((resolve) => {
        recorder.onstop = resolve;
        if (recorder.state !== "inactive") {
          recorder.stop();
        } else {
          resolve();
        }
      });
    }

    async function enterFlashView() {
      const layer = $("flash-layer");
      const preview = $("capture-preview");
      layer.style.backgroundColor = "#000";
      preview.classList.add("active");
      layer.classList.add("active");
      if (layer.requestFullscreen) {
        try {
          await layer.requestFullscreen();
        } catch (error) {
          show(`fullscreen skipped: ${error.message}`);
        }
      }
    }

    async function leaveFlashView() {
      $("flash-layer").style.backgroundColor = "#000";
      $("flash-layer").classList.remove("active");
      $("flash-hud").classList.remove("active");
      $("capture-preview").classList.remove("active");
      if (document.fullscreenElement && document.exitFullscreen) {
        try {
          await document.exitFullscreen();
        } catch (error) {
          /* ignore */
        }
      }
    }

    async function playBrowserFlashTimeline(currentSession) {
      const timeline = currentSession.stimulus.timeline || [];
      const layer = $("flash-layer");
      const hud = $("flash-hud");
      for (let index = 0; index < timeline.length; index += 1) {
        const item = timeline[index];
        layer.style.backgroundColor = rgbCss(item.rgb);
        hud.textContent = `${item.phase}  ${item.color_int}  ${item.start_seconds.toFixed(3)}-${item.end_seconds.toFixed(3)}s`;
        await sleep(Math.max(0, Number(item.duration_seconds || 0) * 1000));
      }
    }

    $("refresh-camera-btn").addEventListener("click", async () => {
      try {
        stopCamera();
        await ensureCamera();
        show("camera ready");
      } catch (error) {
        show(`摄像头失败: ${error}`);
      }
    });

    $("camera-select").addEventListener("change", stopCamera);
    $("facing-mode").addEventListener("change", stopCamera);
    refreshCameras().catch(() => {});

    $("record-btn").addEventListener("click", async () => {
      $("record-btn").disabled = true;
      const chunks = [];
      let recorder = null;
      try {
        const currentSession = await createSession();
        const cameraStream = await ensureCamera();
        const videoTrack = cameraStream.getVideoTracks()[0];
        const mimeType = pickMimeType();
        recorder = new MediaRecorder(cameraStream, mimeType ? { mimeType } : undefined);
        recorder.ondataavailable = (event) => {
          if (event.data && event.data.size) chunks.push(event.data);
        };
        await enterFlashView();
        const startedAt = Date.now();
        recorder.start(250);
        await playBrowserFlashTimeline(currentSession);
        const endedAt = Date.now();
        await stopRecorder(recorder);
        await leaveFlashView();

        const blob = new Blob(chunks, { type: mimeType || "video/webm" });
        const form = new FormData();
        const suffix = blob.type.includes("mp4") ? "mp4" : "webm";
        form.append("file", blob, `recording.${suffix}`);
        form.append("client_metadata", JSON.stringify({
          flash_mode: "browser_timeline",
          mime_type: blob.type,
          bytes: blob.size,
          user_agent: navigator.userAgent,
          requested_total_seconds: currentSession.stimulus.requested_duration_seconds,
          recorded_elapsed_seconds: (endedAt - startedAt) / 1000,
          camera_settings: videoTrack && videoTrack.getSettings ? videoTrack.getSettings() : {},
        }));
        const response = await fetch(`/api/sessions/${currentSession.session_id}/recording`, {
          method: "POST",
          body: form,
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || response.statusText);
        if (data.recording_bundle_url) {
          $("download-recording-link").href = data.recording_bundle_url;
          $("download-recording-link").style.display = "inline-flex";
        }
        show(data);
      } catch (error) {
        if (recorder && recorder.state !== "inactive") {
          await stopRecorder(recorder);
        }
        await leaveFlashView();
        show(`录制失败: ${error}`);
      } finally {
        $("record-btn").disabled = false;
      }
    });
  </script>
</body>
</html>"""


def create_app(args: argparse.Namespace) -> FastAPI:
    output_dir = Path(args.output_dir).resolve()
    ensure_dir(output_dir)
    app = FastAPI(title="Flash Collect Stimulus API")

    def session_dir(session_id: str) -> Path:
        if not session_id or "/" in session_id or "\\" in session_id or ".." in session_id:
            raise HTTPException(status_code=400, detail="invalid_session_id")
        return output_dir / session_id

    def require_safe_file_stem(file_stem: str) -> str:
        if (
            not file_stem
            or ".." in file_stem
            or any(char not in SAFE_FILE_STEM_CHARS for char in file_stem)
        ):
            raise HTTPException(status_code=400, detail="invalid_file_stem")
        return file_stem

    @app.get("/", response_class=HTMLResponse)
    async def home() -> HTMLResponse:
        return HTMLResponse(render_home_page())

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "service": "flash_collect_stimulus_api_service",
            "output_dir": str(output_dir),
            "local_urls": [f"http://{ip}:{args.port}/" for ip in local_ip_candidates()],
            "recommended_total_seconds": RECOMMENDED_TOTAL_SECONDS,
            "duration_policy": "caller_defined_total_seconds",
            "flash_protocol": {
                "name": "fixed_collect_protocol",
                "warmup_seconds": TRAINING_WARMUP_SECONDS,
                "hold_seconds": TRAINING_HOLD_SECONDS,
                "restore_seconds": TRAINING_RESTORE_SECONDS,
                "tail_seconds": TRAINING_TAIL_SECONDS,
                "color_order_rgb": [list(color) for color in TRAINING_COLOR_SEQUENCE_RGB],
                "color_order_packed": [rgb_to_packed_int(color) for color in TRAINING_COLOR_SEQUENCE_RGB],
            },
            "default_palette": {str(key): list(value) for key, value in DEFAULT_PALETTE.items()},
        }

    @app.get("/api/status")
    async def api_status() -> dict[str, Any]:
        return await health()

    @app.post("/api/sessions")
    async def create_session(request: StimulusRequest) -> JSONResponse:
        try:
            colors = validate_request(request)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        session_id = uuid.uuid4().hex[:12]
        current_dir = session_dir(session_id)
        ensure_dir(current_dir)
        metadata_path = current_dir / "metadata.json"
        stimulus = build_stimulus_metadata(request, colors)

        metadata: dict[str, Any] = {
            "session_id": session_id,
            "created_at_unix": time.time(),
            "request": model_to_dict(request),
            "stimulus": stimulus,
            "recordings": [],
        }
        save_json(metadata_path, metadata)

        return JSONResponse(
            {
                "session_id": session_id,
                "metadata_url": f"/api/sessions/{session_id}/metadata",
                "recording_upload_url": f"/api/sessions/{session_id}/recording",
                "stimulus": stimulus,
            }
        )

    @app.get("/api/sessions/{session_id}/metadata")
    async def get_metadata(session_id: str) -> JSONResponse:
        path = session_dir(session_id) / "metadata.json"
        if not path.exists():
            raise HTTPException(status_code=404, detail="metadata_not_found")
        return JSONResponse(load_json(path))

    @app.get("/api/sessions/{session_id}/recordings/{recording_stem}.zip")
    async def get_recording_bundle(session_id: str, recording_stem: str) -> FileResponse:
        recording_stem = require_safe_file_stem(recording_stem)
        current_dir = session_dir(session_id)
        txt_path = current_dir / f"{recording_stem}.txt"
        recording_path = next(
            (
                path
                for path in sorted(current_dir.glob(f"{recording_stem}.*"))
                if path.suffix.lower() in VIDEO_EXTENSIONS
            ),
            None,
        )
        if recording_path is None or not txt_path.exists():
            raise HTTPException(status_code=404, detail="recording_bundle_not_found")
        bundle_path = current_dir / f"{recording_stem}_video_txt.zip"
        try:
            write_bundle_zip(
                bundle_path,
                [
                    (recording_path, f"{session_id}_{recording_path.name}"),
                    (txt_path, f"{session_id}_{txt_path.name}"),
                ],
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=f"recording_bundle_missing_file:{exc}") from exc
        return FileResponse(
            bundle_path,
            media_type="application/zip",
            filename=f"{session_id}_{recording_stem}_video_txt.zip",
        )

    @app.post("/api/sessions/{session_id}/recording")
    async def upload_recording(
        session_id: str,
        file: UploadFile = File(...),
        client_metadata: str = Form(default=""),
    ) -> JSONResponse:
        current_dir = session_dir(session_id)
        metadata_path = current_dir / "metadata.json"
        if not metadata_path.exists():
            raise HTTPException(status_code=404, detail="session_not_found")

        suffix = Path(file.filename or "recording.webm").suffix.lower() or ".webm"
        if suffix not in VIDEO_EXTENSIONS:
            raise HTTPException(status_code=400, detail=f"unsupported_recording_type:{suffix}")

        existing_recordings = [
            path for path in current_dir.glob("recording_*") if path.suffix.lower() in VIDEO_EXTENSIONS
        ]
        recording_index = len(existing_recordings) + 1
        recording_path = current_dir / f"recording_{recording_index:03d}{suffix}"
        while recording_path.exists() or recording_path.with_suffix(".txt").exists():
            recording_index += 1
            recording_path = current_dir / f"recording_{recording_index:03d}{suffix}"
        with recording_path.open("wb") as output:
            shutil.copyfileobj(file.file, output)

        try:
            parsed_client_metadata = json.loads(client_metadata) if client_metadata else {}
        except json.JSONDecodeError:
            parsed_client_metadata = {"raw": client_metadata}

        metadata = load_json(metadata_path)
        try:
            protocol_request = request_from_metadata(metadata)
            colors = validate_request(protocol_request)
            frame_count, fps, used_decode_count = video_metadata(recording_path)
            recording_txt_path = recording_path.with_suffix(".txt")
            write_fixed_color_txt(
                recording_txt_path,
                frame_count=frame_count,
                fps=fps,
                request=protocol_request,
                colors=colors,
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"recording_txt_generation_failed:{exc}") from exc

        recording_protocol = {
            "txt_path": str(recording_txt_path),
            "frame_count": int(frame_count),
            "fps": float(fps),
            "duration_seconds": float(frame_count / fps),
            "used_decode_frame_count": bool(used_decode_count),
            "protocol_mode": "fixed_collect_protocol_recording_duration",
            "warmup_seconds": protocol_request.warmup_seconds,
            "hold_seconds": protocol_request.hold_seconds,
            "restore_seconds": protocol_request.restore_seconds,
            "tail_seconds": protocol_request.tail_seconds,
            "colors_rgb": [list(color) for color in colors],
            "color_ints": [rgb_to_packed_int(color) for color in colors],
        }
        recording_bundle_url = f"/api/sessions/{session_id}/recordings/{recording_path.stem}.zip"
        record = {
            "path": str(recording_path),
            "txt_path": str(recording_txt_path),
            "bundle_url": recording_bundle_url,
            "filename": file.filename,
            "content_type": file.content_type,
            "bytes": recording_path.stat().st_size,
            "txt_bytes": recording_txt_path.stat().st_size,
            "uploaded_at_unix": time.time(),
            "client_metadata": parsed_client_metadata,
            "protocol": recording_protocol,
        }
        metadata.setdefault("recordings", []).append(record)
        save_json(metadata_path, metadata)

        return JSONResponse(
            {
                "session_id": session_id,
                "recording_path": str(recording_path),
                "recording_txt_path": str(recording_txt_path),
                "recording_bundle_url": recording_bundle_url,
                "recording_bytes": record["bytes"],
                "recording_txt_bytes": record["txt_bytes"],
                "recording_protocol": recording_protocol,
                "metadata_url": f"/api/sessions/{session_id}/metadata",
            }
        )

    return app


def main() -> None:
    args = parse_args()
    app = create_app(args)
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        ssl_certfile=args.ssl_certfile,
        ssl_keyfile=args.ssl_keyfile,
    )


if __name__ == "__main__":
    main()
