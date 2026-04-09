"""
ThunderGuard ONNX automated test script.

This script is designed for the user's `anti-spoofing_scc_175` environment and
focuses on four things:

1. Export one or more PyTorch checkpoints to ONNX.
2. Run PyTorch inference and ONNXRuntime inference on the same sampled data.
3. Compare the outputs with simple, explicit metrics.
4. Write a text report that explains:
   - test method
   - expected data format
   - metrics
   - pass/fail basis

Recommended environment:
    /home/scc/anaconda3/envs/anti-spoofing_scc_175/bin/python
"""

import argparse
import io
import json
import os
import sys
from contextlib import redirect_stdout
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import onnxruntime
import torch

try:
    import onnx
except ImportError:
    onnx = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTG_ROOT = PROJECT_ROOT / "pytg"
sys.path.insert(0, str(PYTG_ROOT))

import networks  # noqa: E402
from networks import load_checkpoint  # noqa: E402


@dataclass
class SampleData:
    txt_name: str
    txt_path: Path
    image_path: Path
    colors: np.ndarray
    normalized_input: np.ndarray
    color_flags: np.ndarray


@dataclass
class ModelTestResult:
    checkpoint_path: str
    export_file: str
    export_success: bool
    onnx_check_success: bool
    onnx_check_message: str
    pytorch_score: List[float]
    onnx_score: List[float]
    pytorch_color: List[List[float]]
    onnx_color: List[List[float]]
    max_abs_diff_score: float
    max_abs_diff_color: float
    pytorch_color_pass: bool
    onnx_color_pass: bool
    decision_pass: bool
    decision_reason: str


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export ThunderGuard checkpoints to ONNX and compare PyTorch vs ONNXRuntime outputs."
    )
    parser.add_argument(
        "--checkpoint-paths",
        nargs="+",
        required=True,
        help="One or more .pth.tar checkpoints to test.",
    )
    parser.add_argument(
        "--sample-dir",
        required=True,
        help="ThunderGuard sample directory containing matching .txt and .jpg files.",
    )
    parser.add_argument(
        "--export-dir",
        required=True,
        help="Directory where exported ONNX files will be written.",
    )
    parser.add_argument(
        "--report-txt",
        required=True,
        help="Output text report path.",
    )
    parser.add_argument(
        "--sample-count",
        type=int,
        default=1,
        help="How many samples to test. Default is 1 for a fast, stable regression check.",
    )
    parser.add_argument(
        "--network",
        default="MoEA",
        help="ThunderGuard network module name.",
    )
    parser.add_argument(
        "--model",
        default="MoEA",
        help="ThunderGuard model class name.",
    )
    parser.add_argument(
        "--infer-type",
        default="score",
        help="Inference type used during ONNX export. Usually score.",
    )
    parser.add_argument(
        "--score-diff-threshold",
        type=float,
        default=1e-4,
        help="Maximum allowed absolute difference between PyTorch and ONNX scores.",
    )
    parser.add_argument(
        "--color-diff-threshold",
        type=float,
        default=1e-4,
        help="Maximum allowed absolute difference between PyTorch and ONNX color outputs.",
    )
    return parser.parse_args()


def collect_sample_names(sample_dir: Path, sample_count: int) -> List[str]:
    names = []
    for txt_path in sorted(sample_dir.glob("*.txt")):
        jpg_path = txt_path.with_suffix(".jpg")
        if jpg_path.exists():
            names.append(txt_path.name)
        if len(names) >= sample_count:
            break
    if not names:
        raise RuntimeError(f"no valid ThunderGuard samples found in {sample_dir}")
    return names


def read_colors(txt_path: Path) -> np.ndarray:
    colors = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            packed = int(line.strip())
            r = (packed & 0x00FF0000) >> 16
            g = (packed & 0x0000FF00) >> 8
            b = packed & 0x000000FF
            colors.append((r, g, b))
    return np.array(colors)


def load_stacked_thunderguard_image(image_path: Path) -> np.ndarray:
    """
    ThunderGuard stores 6 cue images vertically in a single JPG.

    Each cue is cropped to the training/inference layout:
    - crop border
    - shape -> (3, 256, 256)
    - stack -> (6, 3, 256, 256)
    """
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"failed to read image: {image_path}")

    cues = []
    for row in range(3):
        for col in range(2):
            cue_id = row * 2 + col
            cue = image[cue_id * 288 + 16:(cue_id + 1) * 288 - 16, 16:288 - 16, :]
            cue = cue.transpose((2, 0, 1)).astype(np.float32)
            cues.append(cue.reshape(1, 3, 256, 256))
    return np.concatenate(cues, axis=0)


def build_color_flags(colors: np.ndarray) -> np.ndarray:
    flags = []
    for i in range(3):
        relation = np.where(colors[i] > colors[i, [1, 2, 0]], 1, 0) + np.where(colors[i] < colors[i, [1, 2, 0]], -1, 0)
        flags.append(relation)
        flags.append(relation)
    return np.array(flags, dtype=np.int32)


def load_sample(sample_dir: Path, txt_name: str) -> SampleData:
    txt_path = sample_dir / txt_name
    image_path = txt_path.with_suffix(".jpg")
    colors = read_colors(txt_path)
    stacked = load_stacked_thunderguard_image(image_path)
    normalized_input = ((stacked - 127.5) / 128.0).astype(np.float32)
    return SampleData(
        txt_name=txt_name,
        txt_path=txt_path,
        image_path=image_path,
        colors=colors,
        normalized_input=normalized_input,
        color_flags=build_color_flags(colors),
    )


def validate_color_prediction(color_pred: np.ndarray, color_flags: np.ndarray) -> bool:
    """
    Reuse the repository's color-consistency idea:
    if both duplicated checks inside a pair fail, the sample is considered invalid.
    """
    color_pred = np.reshape(color_pred, [-1, 2, 3])
    color_flags = np.reshape(color_flags, [-1, 2, 3])

    for pred_pair, flag_pair in zip(color_pred, color_flags):
        pair_failures = 0
        for idx in range(2):
            pred = pred_pair[idx]
            flag = flag_pair[idx]
            passed = float(np.min(flag * (pred - 0.5))) >= 0
            if not passed:
                pair_failures += 1
        if pair_failures == 2:
            return False
    return True


def create_model(network_name: str, model_name: str, infer_type: str):
    network_module = getattr(networks, network_name)
    return getattr(network_module, model_name)(infer_type=infer_type)


def load_model_weights(model, checkpoint_path: str):
    # The original load_checkpoint prints a large amount of "copy value to ..."
    # noise. We silence that so the automated report stays readable.
    with redirect_stdout(io.StringIO()):
        checkpoint = load_checkpoint(checkpoint_path, model)
    return checkpoint


def export_to_onnx(model, export_file: Path, device: torch.device, infer_type: str):
    if onnx is None:
        raise RuntimeError(
            "onnx package is required for torch.onnx.export in this environment. "
            "If you are using anti-spoofing_scc_175, install it first, for example:\n"
            "env -u ALL_PROXY -u all_proxy -u HTTP_PROXY -u HTTPS_PROXY -u http_proxy -u https_proxy "
            "/home/scc/anaconda3/envs/anti-spoofing_scc_175/bin/pip install onnx"
        )
    export_file.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    model = model.to(device)
    dummy_input = torch.randn(6, 3, 256, 256, device=device)

    if infer_type == "score":
        torch.onnx.export(
            model,
            dummy_input,
            str(export_file),
            export_params=True,
            verbose=False,
            input_names=["input"],
            output_names=["color", "score"],
            keep_initializers_as_inputs=True,
            opset_version=11,
        )
    else:
        torch.onnx.export(
            model,
            dummy_input,
            str(export_file),
            export_params=True,
            verbose=False,
            input_names=["input"],
            output_names=["map"],
            keep_initializers_as_inputs=True,
            opset_version=11,
        )


def run_pytorch(model, normalized_input: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        color_pred, score_pred = model(torch.from_numpy(normalized_input).to(device))
    return color_pred.detach().cpu().numpy(), score_pred.detach().cpu().numpy()


def run_onnxruntime(onnx_file: Path, normalized_input: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    session = onnxruntime.InferenceSession(str(onnx_file), providers=["CPUExecutionProvider"])
    outputs = session.run(None, {"input": normalized_input})
    if len(outputs) != 2:
        raise RuntimeError(f"expected 2 ONNX outputs, got {len(outputs)}")
    return outputs[0], outputs[1]


def maybe_check_onnx(onnx_file: Path) -> Tuple[bool, str]:
    if onnx is None:
        return False, "onnx package not installed in this environment; structural checker skipped"
    try:
        model_proto = onnx.load(str(onnx_file))
        onnx.checker.check_model(model_proto)
        return True, "onnx.checker.check_model passed"
    except Exception as e:
        return False, f"onnx structural check failed: {type(e).__name__}: {e}"


def safe_name_from_checkpoint(checkpoint_path: str) -> str:
    path = Path(checkpoint_path)
    grand_parent = path.parent.parent.name
    parent = path.parent.name
    stem = path.stem.replace(".pth", "")
    return f"{grand_parent}_{parent}_{stem}"


def compare_model_on_sample(
    checkpoint_path: str,
    sample: SampleData,
    export_dir: Path,
    network_name: str,
    model_name: str,
    infer_type: str,
    score_threshold: float,
    color_threshold: float,
) -> ModelTestResult:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = create_model(network_name, model_name, infer_type)
    load_model_weights(model, checkpoint_path)

    export_file = export_dir / f"{safe_name_from_checkpoint(checkpoint_path)}_{infer_type}.onnx"
    export_to_onnx(model, export_file, device, infer_type)
    onnx_check_success, onnx_check_message = maybe_check_onnx(export_file)

    pytorch_color, pytorch_score = run_pytorch(model.to(device), sample.normalized_input)
    onnx_color, onnx_score = run_onnxruntime(export_file, sample.normalized_input)

    max_abs_diff_score = float(np.max(np.abs(np.reshape(pytorch_score, [-1]) - np.reshape(onnx_score, [-1]))))
    max_abs_diff_color = float(np.max(np.abs(np.reshape(pytorch_color, [-1]) - np.reshape(onnx_color, [-1]))))

    pytorch_color_pass = validate_color_prediction(pytorch_color, sample.color_flags)
    onnx_color_pass = validate_color_prediction(onnx_color, sample.color_flags)

    decision_pass = (
        max_abs_diff_score <= score_threshold
        and max_abs_diff_color <= color_threshold
        and pytorch_color_pass == onnx_color_pass
    )

    reasons = []
    if max_abs_diff_score > score_threshold:
        reasons.append(f"score diff {max_abs_diff_score:.8f} > {score_threshold:.8f}")
    if max_abs_diff_color > color_threshold:
        reasons.append(f"color diff {max_abs_diff_color:.8f} > {color_threshold:.8f}")
    if pytorch_color_pass != onnx_color_pass:
        reasons.append("PyTorch and ONNX color-validation decisions differ")
    if not reasons:
        reasons.append("all comparison metrics are within thresholds")

    return ModelTestResult(
        checkpoint_path=checkpoint_path,
        export_file=str(export_file),
        export_success=True,
        onnx_check_success=onnx_check_success,
        onnx_check_message=onnx_check_message,
        pytorch_score=np.round(np.reshape(pytorch_score, [-1]), 6).tolist(),
        onnx_score=np.round(np.reshape(onnx_score, [-1]), 6).tolist(),
        pytorch_color=np.round(np.reshape(pytorch_color, [-1, 3]), 6).tolist(),
        onnx_color=np.round(np.reshape(onnx_color, [-1, 3]), 6).tolist(),
        max_abs_diff_score=max_abs_diff_score,
        max_abs_diff_color=max_abs_diff_color,
        pytorch_color_pass=bool(pytorch_color_pass),
        onnx_color_pass=bool(onnx_color_pass),
        decision_pass=bool(decision_pass),
        decision_reason="; ".join(reasons),
    )


def format_report(
    args,
    env_python: str,
    sample_names: List[str],
    samples: List[SampleData],
    results: Dict[str, ModelTestResult],
) -> str:
    lines = []
    lines.append("ThunderGuard ONNX Automated Test Report")
    lines.append(f"time: {datetime.now().isoformat()}")
    lines.append("")
    lines.append("Environment")
    lines.append(f"- python: {env_python}")
    lines.append(f"- torch: {torch.__version__}")
    lines.append(f"- onnxruntime: {onnxruntime.__version__}")
    lines.append(f"- cv2: {cv2.__version__}")
    lines.append(f"- numpy: {np.__version__}")
    lines.append(f"- onnx: {getattr(onnx, '__version__', 'not installed') if onnx is not None else 'not installed'}")
    lines.append("")
    lines.append("Test Method")
    lines.append("- Export each checkpoint to ONNX with infer_type=score.")
    lines.append("- Run the same sampled input through the original PyTorch model and the exported ONNX model.")
    lines.append("- Compare score output and color-head output with max absolute difference.")
    lines.append("- Re-run the repository color-validation rule on both outputs to check decision consistency.")
    lines.append("")
    lines.append("Test Data Format")
    lines.append("- sample_dir must contain matching .txt and .jpg files.")
    lines.append("- each .jpg is a vertically stacked ThunderGuard image containing 6 cue images.")
    lines.append("- each .txt stores packed flash colors used to build the color-consistency rule.")
    lines.append("- optional _d.jpg files may exist in the directory but are not required for this score-model test.")
    lines.append("")
    lines.append("Test Metrics")
    lines.append(f"- max_abs_diff_score: must be <= {args.score_diff_threshold}")
    lines.append(f"- max_abs_diff_color: must be <= {args.color_diff_threshold}")
    lines.append("- color_validation_consistency: PyTorch and ONNX must produce the same pass/fail result")
    lines.append("")
    lines.append("Decision Basis")
    lines.append("- PASS: export succeeds, ONNXRuntime inference succeeds, diffs stay within thresholds, and color-validation decisions match.")
    lines.append("- FAIL: any export/inference step fails, diffs exceed thresholds, or color-validation decisions diverge.")
    lines.append("")
    lines.append("Selected Samples")
    lines.append(json.dumps(sample_names, ensure_ascii=False, indent=2))
    lines.append("")

    for sample in samples:
        lines.append(f"Sample: {sample.txt_name}")
        lines.append(f"- txt_path: {sample.txt_path}")
        lines.append(f"- image_path: {sample.image_path}")
        lines.append(f"- color_triplets: {sample.colors.tolist()}")
        lines.append(
            "- input_stats: "
            + json.dumps(
                {
                    "min": float(sample.normalized_input.min()),
                    "max": float(sample.normalized_input.max()),
                    "mean": float(sample.normalized_input.mean()),
                    "std": float(sample.normalized_input.std()),
                },
                ensure_ascii=False,
            )
        )
        lines.append(f"- input_first_12_values: {np.round(sample.normalized_input[0].reshape(-1)[:12], 6).tolist()}")
        lines.append("")

        for checkpoint_path, result in results.items():
            lines.append(f"Checkpoint: {checkpoint_path}")
            lines.append(f"- export_file: {result.export_file}")
            lines.append(f"- export_success: {result.export_success}")
            lines.append(f"- onnx_check_success: {result.onnx_check_success}")
            lines.append(f"- onnx_check_message: {result.onnx_check_message}")
            lines.append(f"- pytorch_score: {result.pytorch_score}")
            lines.append(f"- onnx_score: {result.onnx_score}")
            lines.append(f"- pytorch_color_pass: {result.pytorch_color_pass}")
            lines.append(f"- onnx_color_pass: {result.onnx_color_pass}")
            lines.append(f"- max_abs_diff_score: {result.max_abs_diff_score:.10f}")
            lines.append(f"- max_abs_diff_color: {result.max_abs_diff_color:.10f}")
            lines.append(f"- decision_pass: {result.decision_pass}")
            lines.append(f"- decision_reason: {result.decision_reason}")
            lines.append("")

        break

    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    sample_dir = Path(args.sample_dir)
    export_dir = Path(args.export_dir)
    report_txt = Path(args.report_txt)

    sample_names = collect_sample_names(sample_dir, args.sample_count)
    samples = [load_sample(sample_dir, name) for name in sample_names]

    # For a regression-style automated check we test the same sampled input across
    # all checkpoints, which makes output differences easier to interpret.
    primary_sample = samples[0]

    results: Dict[str, ModelTestResult] = {}
    for checkpoint_path in args.checkpoint_paths:
        result = compare_model_on_sample(
            checkpoint_path=checkpoint_path,
            sample=primary_sample,
            export_dir=export_dir,
            network_name=args.network,
            model_name=args.model,
            infer_type=args.infer_type,
            score_threshold=args.score_diff_threshold,
            color_threshold=args.color_diff_threshold,
        )
        results[checkpoint_path] = result

    env_python = sys.executable
    report_txt.parent.mkdir(parents=True, exist_ok=True)
    report = format_report(args, env_python, sample_names, samples, results)
    report_txt.write_text(report, encoding="utf-8")
    print(f"report_written: {report_txt}")


if __name__ == "__main__":
    main()
