import argparse
import base64
import html
from typing import Optional

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse

from fused_face_liveness_api import (
    FusionLivenessService,
    decode_base64_image,
    extract_request_payload,
    read_image_from_bytes,
    read_image_from_path,
    read_image_from_url,
    resolve_default_gallery_dir,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="FastAPI fused face liveness service V2 with simple upload UI."
    )
    parser.add_argument("--host", default="0.0.0.0", help="Bind host.")
    parser.add_argument("--port", type=int, default=18120, help="Bind port.")
    parser.add_argument(
        "--gallery-dir",
        default=str(resolve_default_gallery_dir()),
        help="Bank folder with labeled real/head-model reference images.",
    )
    parser.add_argument("--fr-threshold", type=float, default=0.49, help="Bank FR match threshold.")
    parser.add_argument(
        "--spoof-threshold",
        type=float,
        default=0.68,
        help="Fallback DeePixBiS threshold. If bank is not matched and pixel_score > 0.68, output real.",
    )
    parser.add_argument("--yolo-path", default="/supercloud/llm-code/scc/scc/Liveness_Detection/yolov7_face/yolov7-w6-face.pt")
    parser.add_argument("--arcface-path", default="/supercloud/llm-code/scc/scc/Liveness_Detection/model_16.pt")
    parser.add_argument(
        "--deepixbis-weights",
        default="/supercloud/llm-code/scc/scc/Liveness_Detection/Face-Anti-Spoofing-using-DeePixBiS/DeePixBiS.pth",
    )
    parser.add_argument("--device", default="cuda", help="Torch device, for example cuda:0 or cpu.")
    parser.add_argument(
        "--output-dir",
        default="/supercloud/llm-code/scc/scc/Liveness_Detection/Face-Anti-Spoofing-using-DeePixBiS/fusion_api_outputs_v2",
    )
    return parser.parse_args()


def page_shell(content: str):
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Face Liveness API V2</title>
  <style>
    :root {{
      --bg: #f4efe6;
      --panel: #fffaf2;
      --ink: #1d1b18;
      --muted: #6e6457;
      --line: #d8c7ad;
      --ok: #1f7a4d;
      --bad: #ad2e24;
      --soft: #efe2cc;
      --accent: #b8702a;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Source Han Sans SC", "Noto Sans SC", "PingFang SC", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, #fff7df 0, transparent 32%),
        radial-gradient(circle at bottom right, #f0dcc5 0, transparent 28%),
        linear-gradient(135deg, #f3ecdf 0%, #ead9c0 100%);
      min-height: 100vh;
    }}
    .wrap {{
      max-width: 1100px;
      margin: 0 auto;
      padding: 32px 20px 48px;
    }}
    .hero {{
      display: grid;
      grid-template-columns: 1.2fr 0.8fr;
      gap: 18px;
      margin-bottom: 20px;
    }}
    .card {{
      background: rgba(255,250,242,0.92);
      border: 1px solid var(--line);
      border-radius: 20px;
      padding: 22px;
      box-shadow: 0 12px 40px rgba(92, 63, 28, 0.08);
      backdrop-filter: blur(10px);
    }}
    h1 {{ margin: 0 0 10px; font-size: 30px; line-height: 1.15; }}
    h2 {{ margin: 0 0 12px; font-size: 20px; }}
    p {{ margin: 0 0 10px; color: var(--muted); line-height: 1.6; }}
    .badge {{
      display: inline-block;
      padding: 6px 10px;
      border-radius: 999px;
      background: var(--soft);
      color: #6d4219;
      font-size: 13px;
      margin-right: 8px;
      margin-bottom: 8px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 18px;
    }}
    form {{ display: grid; gap: 12px; }}
    label {{ font-weight: 700; font-size: 14px; }}
    input[type="file"], input[type="text"], textarea {{
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 12px 14px;
      background: #fffdf9;
      font: inherit;
    }}
    textarea {{ min-height: 120px; resize: vertical; }}
    button {{
      border: 0;
      border-radius: 14px;
      padding: 12px 16px;
      font: inherit;
      font-weight: 700;
      color: white;
      background: linear-gradient(135deg, #b7702c 0%, #8d4f17 100%);
      cursor: pointer;
    }}
    .result-ok {{ color: var(--ok); }}
    .result-bad {{ color: var(--bad); }}
    .mono {{
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 13px;
      word-break: break-all;
    }}
    .imgbox img {{
      width: 100%;
      border-radius: 16px;
      border: 1px solid var(--line);
      background: white;
    }}
    ul {{ margin: 10px 0 0; padding-left: 18px; }}
    li {{ margin-bottom: 8px; color: var(--muted); }}
    @media (max-width: 860px) {{
      .hero, .grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    {content}
  </div>
</body>
</html>"""


def render_home(args):
    return page_shell(
        f"""
        <div class="hero">
          <section class="card">
            <h1>Face Liveness API V2</h1>
            <p>同一套服务同时提供 JSON 接口和网页上传界面。网页和接口都使用相对路径，不依赖固定 IP，所以无论你通过本机地址、局域网地址还是域名访问，前端都会自动请求当前站点。</p>
            <span class="badge">Bank FR &gt; 0.49</span>
            <span class="badge">DeePixBiS &gt; 0.68 fallback</span>
            <span class="badge">Simple Web UI</span>
          </section>
          <section class="card">
            <h2>当前规则</h2>
            <p>1. 先匹配 bank。</p>
            <p>2. 命中 bank 且 FR 分数大于 0.49，按 bank 标签判真人/假人。</p>
            <p>3. 没命中 bank 时，DeePixBiS 分数大于 0.68 判真人，否则判假人。</p>
            <p class="mono">bank_dir: {html.escape(str(args.gallery_dir))}</p>
          </section>
        </div>
        <div class="grid">
          <section class="card">
            <h2>网页上传测试</h2>
            <form action="/predict-ui" method="post" enctype="multipart/form-data">
              <div>
                <label for="file">上传图片文件</label>
                <input id="file" name="file" type="file" accept="image/*" required>
              </div>
              <button type="submit">上传并识别</button>
            </form>
          </section>
          <section class="card">
            <h2>JSON 接口</h2>
            <p>接口地址使用当前访问站点的相对路径 `/predict`，前端不写死任何 IP。</p>
            <div class="mono">POST /predict</div>
            <ul>
              <li>推荐外部调用方式：文件上传、base64、图片 URL</li>
              <li>`image_path` 只适合服务所在机器本地调用</li>
              <li>结果字段重点看 `binary_result`、`fr.min_score`、`spoof.pixel_score`、`basis`</li>
            </ul>
          </section>
        </div>
        """
    )


def render_result(result, uploaded_name: str, upload_preview_b64: Optional[str]):
    binary_result = result.get("binary_result", "unknown")
    result_class = "result-ok" if binary_result == "real" else "result-bad"
    basis_items = "".join(
        f"<li>{html.escape(str(item))}</li>" for item in result.get("basis", [])
    )
    preview_block = ""
    if upload_preview_b64:
        preview_block = f"""
        <section class="card imgbox">
          <h2>上传原图</h2>
          <img src="data:image/jpeg;base64,{upload_preview_b64}" alt="uploaded-image">
        </section>
        """

    annotated_b64 = result.get("annotated_image_base64")
    aligned_b64 = result.get("aligned_face_base64")
    annotated_block = ""
    if annotated_b64:
        annotated_block = f"""
        <section class="card imgbox">
          <h2>结果绘框图</h2>
          <img src="data:image/jpeg;base64,{annotated_b64}" alt="annotated-image">
        </section>
        """
    aligned_block = ""
    if aligned_b64:
        aligned_block = f"""
        <section class="card imgbox">
          <h2>对齐人脸</h2>
          <img src="data:image/jpeg;base64,{aligned_b64}" alt="aligned-face">
        </section>
        """

    return page_shell(
        f"""
        <div class="hero">
          <section class="card">
            <h1>识别完成</h1>
            <p>输入文件：<span class="mono">{html.escape(uploaded_name)}</span></p>
            <p class="{result_class}"><strong>最终结果：{html.escape(str(result.get("binary_result")))}</strong></p>
            <p>详细类别：<span class="mono">{html.escape(str(result.get("result")))}</span></p>
            <p>FR min_score：<span class="mono">{html.escape(str(result.get("fr", {}).get("min_score")))}</span></p>
            <p>DeePixBiS pixel_score：<span class="mono">{html.escape(str(result.get("spoof", {}).get("pixel_score")))}</span></p>
            <p>最佳 bank 匹配：<span class="mono">{html.escape(str(result.get("best_bank_match")))}</span></p>
          </section>
          <section class="card">
            <h2>判断依据</h2>
            <ul>{basis_items}</ul>
          </section>
        </div>
        <div class="grid">
          {preview_block}
          {annotated_block}
          {aligned_block}
        </div>
        <div style="margin-top:18px">
          <a href="/" style="color:#8d4f17;font-weight:700;text-decoration:none;">返回上传页</a>
        </div>
        """
    )


def create_app(args):
    app = FastAPI(title="Fused Face Liveness API V2", version="2.0.0")
    service = FusionLivenessService(args)

    @app.get("/", response_class=HTMLResponse)
    async def home():
        return render_home(args)

    @app.get("/ui", response_class=HTMLResponse)
    async def ui():
        return render_home(args)

    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "bank_dir": str(service.bank_dir),
            "bank_size": len(service.bank_records),
            "fr_threshold": args.fr_threshold,
            "spoof_threshold": args.spoof_threshold,
            "version": "v2",
        }

    @app.post("/predict")
    async def predict(
        request: Request,
        file: Optional[UploadFile] = File(default=None),
        image_base64: Optional[str] = Form(default=None),
        image_path: Optional[str] = Form(default=None),
        image_url: Optional[str] = Form(default=None),
        return_images: Optional[bool] = Form(default=False),
    ):
        payload = await extract_request_payload(request)

        if not image_base64:
            image_base64 = payload.get("image_base64") or request.query_params.get("image_base64")
        if not image_path:
            image_path = payload.get("image_path") or request.query_params.get("image_path")
        if not image_url:
            image_url = payload.get("image_url") or request.query_params.get("image_url")
        if "return_images" in payload and payload.get("return_images") is not None:
            return_images = bool(payload.get("return_images"))

        source_type = None
        source_value = None
        image_bgr = None

        if file is not None:
            source_type = "upload_file"
            source_value = file.filename or "uploaded_image"
            image_bgr = read_image_from_bytes(await file.read())
        elif payload.get("raw_bytes"):
            source_type = "raw_bytes"
            source_value = "request_body_image"
            image_bgr = read_image_from_bytes(payload["raw_bytes"])
        elif image_base64:
            source_type = "base64"
            source_value = "inline_base64"
            image_bgr = decode_base64_image(image_base64)
        elif image_path:
            source_type = "path"
            source_value = image_path
            image_bgr = read_image_from_path(image_path)
        elif image_url:
            source_type = "url"
            source_value = image_url
            image_bgr = read_image_from_url(image_url)

        if image_bgr is None:
            raise HTTPException(status_code=400, detail="No valid image input found.")

        try:
            result = service.predict(
                image_bgr=image_bgr,
                source_type=source_type or "unknown",
                source_value=source_value or "unknown",
                return_images=bool(return_images),
            )
            return JSONResponse(content=result)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/predict-ui", response_class=HTMLResponse)
    async def predict_ui(file: UploadFile = File(...)):
        image_bytes = await file.read()
        image_bgr = read_image_from_bytes(image_bytes)
        if image_bgr is None:
            raise HTTPException(status_code=400, detail="image_decode_failed")
        result = service.predict(
            image_bgr=image_bgr,
            source_type="upload_file",
            source_value=file.filename or "uploaded_image",
            return_images=True,
        )
        upload_preview_b64 = base64.b64encode(image_bytes).decode("utf-8")
        return render_result(result, file.filename or "uploaded_image", upload_preview_b64)

    return app


def main():
    args = parse_args()
    app = create_app(args)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
