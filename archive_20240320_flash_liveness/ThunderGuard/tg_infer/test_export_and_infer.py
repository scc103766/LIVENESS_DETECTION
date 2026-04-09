import argparse
import os
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import onnx
import torch
from onnx.reference import ReferenceEvaluator

try:
    import onnxruntime
except ImportError:
    onnxruntime = None

import sys

PYTG_ROOT = Path(__file__).resolve().parents[1] / "pytg"
sys.path.insert(0, str(PYTG_ROOT))

import networks  # noqa: E402
from networks import load_checkpoint  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Export checkpoints to ONNX and run sampled inference.")
    parser.add_argument(
        "--checkpoint-paths",
        nargs="+",
        required=True,
        help="Checkpoint files to test",
    )
    parser.add_argument(
        "--sample-dir",
        required=True,
        help="Directory containing ThunderGuard-style .jpg/.txt/_d.jpg samples",
    )
    parser.add_argument(
        "--sample-count",
        type=int,
        default=3,
        help="How many samples to record in detail",
    )
    parser.add_argument(
        "--report-txt",
        required=True,
        help="Output txt report path",
    )
    parser.add_argument(
        "--export-dir",
        required=True,
        help="Output directory for exported onnx files",
    )
    parser.add_argument(
        "--network",
        default="MoEA",
        help="Network name",
    )
    parser.add_argument(
        "--model",
        default="MoEA",
        help="Model class name",
    )
    parser.add_argument(
        "--infer-type",
        default="score",
        help="Infer type for ONNX export",
    )
    return parser.parse_args()


class ONNXModel:
    def __init__(self, onnx_path):
        self.backend = None
        self.input_names = []
        self.output_names = []
        self.session = None
        self.net = None
        self.reference_session = None

        if onnxruntime is not None:
            self.session = onnxruntime.InferenceSession(str(onnx_path))
            self.input_names = [node.name for node in self.session.get_inputs()]
            self.output_names = [node.name for node in self.session.get_outputs()]
            self.backend = "onnxruntime"
        else:
            try:
                model_proto = onnx.load(str(onnx_path))
                self.reference_session = ReferenceEvaluator(model_proto)
                self.input_names = [i.name for i in model_proto.graph.input]
                self.output_names = [o.name for o in model_proto.graph.output]
                self.backend = "onnx.reference"
            except Exception:
                self.net = cv2.dnn.readNetFromONNX(str(onnx_path))
                self.output_names = self.net.getUnconnectedOutLayersNames()
                self.backend = "cv2.dnn"

    def forward(self, tensor):
        if self.backend == "onnxruntime":
            return self.session.run(self.output_names, {self.input_names[0]: tensor})
        if self.backend == "onnx.reference":
            return self.reference_session.run(None, {self.input_names[0]: tensor})
        self.net.setInput(tensor)
        return self.net.forward(self.output_names)


def read_colors(file_path):
    values = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            color = int(line.strip())
            c1 = (color & 0x00FF0000) >> 16
            c2 = (color & 0x0000FF00) >> 8
            c3 = color & 0x000000FF
            values.append((c1, c2, c3))
    return values


def only_load_image(image_path):
    img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"failed to read image: {image_path}")
    normal_cue_list = []
    for i in range(3):
        for j in range(2):
            nid = i * 2 + j
            cur_img = img[nid * 288 + 16:(nid + 1) * 288 - 16, 16:288 - 16, :].transpose((2, 0, 1)).astype(np.float32)
            normal_cue_list.append(cur_img.reshape(1, 3, 256, 256))
    return np.concatenate(normal_cue_list, axis=0)


def load_sample(sample_dir, txt_name):
    txt_path = Path(sample_dir) / txt_name
    image_path = txt_path.with_suffix(".jpg")
    colors = np.array(read_colors(txt_path))
    normal_cues = only_load_image(image_path)

    sc_flag = []
    for i in range(3):
        sc_flag.append(np.where(colors[i] > colors[i, [1, 2, 0]], 1, 0) + np.where(colors[i] < colors[i, [1, 2, 0]], -1, 0))
        sc_flag.append(sc_flag[-1])
    return {
        "txt_name": txt_name,
        "txt_path": str(txt_path),
        "image_path": str(image_path),
        "colors": colors,
        "normal_cues": normal_cues,
        "normalized_input": ((normal_cues - 127.5) / 128.0).astype(np.float32),
        "sc_flag": np.array(sc_flag, dtype=np.int32),
    }


def validate_color(color_pred, sc_flag):
    color_pred = np.reshape(color_pred, [-1, 2, 3])
    sc_flag = np.reshape(sc_flag, [-1, 2, 3])
    flag = 1
    pair_details = []
    for pair_idx, (pred_pair, flag_pair) in enumerate(zip(color_pred, sc_flag)):
        error_time = 0
        checks = []
        for idx in range(2):
            cur_pred = pred_pair[idx]
            cur_flag = flag_pair[idx]
            passed = float(np.min(cur_flag * (cur_pred - 0.5))) >= 0
            if not passed:
                error_time += 1
            checks.append(
                {
                    "slot": idx,
                    "pred": np.round(cur_pred, 6).tolist(),
                    "flag": cur_flag.tolist(),
                    "passed": bool(passed),
                }
            )
        if error_time == 2:
            flag = 0
        pair_details.append({"pair_idx": pair_idx, "checks": checks, "pair_passed": error_time < 2})
        if flag == 0:
            break
    return flag, pair_details


def export_checkpoint(model, checkpoint_path, export_file, device, infer_type):
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


def infer_with_torch(model, input_tensor):
    with torch.no_grad():
        device = next(model.parameters()).device
        color_pred, score_pred = model(torch.from_numpy(input_tensor).to(device))
    return color_pred.detach().cpu().numpy(), score_pred.detach().cpu().numpy()


def collect_samples(sample_dir, sample_count):
    names = []
    for txt_name in sorted(os.listdir(sample_dir)):
        if not txt_name.endswith(".txt"):
            continue
        base = txt_name[:-4]
        if os.path.exists(os.path.join(sample_dir, base + ".jpg")):
            names.append(txt_name)
        if len(names) >= sample_count:
            break
    return names


def write_report(report_path, lines):
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    export_dir = Path(args.export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    sample_names = collect_samples(args.sample_dir, args.sample_count)
    samples = [load_sample(args.sample_dir, name) for name in sample_names]

    report_lines = []
    report_lines.append("ThunderGuard checkpoint -> ONNX -> infer test report")
    report_lines.append(f"time: {datetime.now().isoformat()}")
    report_lines.append(f"device: {device}")
    report_lines.append(f"onnxruntime_available: {onnxruntime is not None}")
    report_lines.append(f"opencv_version: {cv2.__version__}")
    report_lines.append(f"sample_dir: {args.sample_dir}")
    report_lines.append(f"sample_names: {sample_names}")
    report_lines.append("")

    for checkpoint_path in args.checkpoint_paths:
        checkpoint_path = Path(checkpoint_path)
        model_name = checkpoint_path.stem.replace(".pth", "")
        export_file = export_dir / f"{checkpoint_path.parent.name}_{model_name}_{args.infer_type}.onnx"

        network_module = getattr(networks, args.network)
        model = getattr(network_module, args.model)(infer_type=args.infer_type)
        checkpoint = load_checkpoint(str(checkpoint_path), model)
        model.eval()
        model = model.to(device)

        export_checkpoint(model, checkpoint_path, export_file, device, args.infer_type)
        onnx_model = onnx.load(str(export_file))
        onnx.checker.check_model(onnx_model)
        ort_worker = ONNXModel(export_file)

        report_lines.append(f"=== checkpoint: {checkpoint_path} ===")
        report_lines.append(f"checkpoint_keys: {sorted(checkpoint.keys())}")
        report_lines.append(f"export_file: {export_file}")
        report_lines.append(f"export_file_size_bytes: {export_file.stat().st_size}")
        report_lines.append(f"onnx_graph_inputs: {[node.name for node in onnx_model.graph.input]}")
        report_lines.append(f"onnx_graph_outputs: {[node.name for node in onnx_model.graph.output]}")
        report_lines.append(f"onnx_infer_backend: {ort_worker.backend}")

        for sample in samples:
            input_tensor = sample["normalized_input"]
            torch_color, torch_score = infer_with_torch(model, input_tensor)
            torch_flag, torch_pairs = validate_color(torch_color, sample["sc_flag"])

            report_lines.append(f"sample: {sample['txt_name']}")
            report_lines.append(f"  image_path: {sample['image_path']}")
            report_lines.append(f"  raw_color_triplets: {sample['colors'].tolist()}")
            report_lines.append(f"  input_shape: {list(input_tensor.shape)}")
            report_lines.append(
                "  input_stats: min={:.6f}, max={:.6f}, mean={:.6f}, std={:.6f}".format(
                    float(input_tensor.min()),
                    float(input_tensor.max()),
                    float(input_tensor.mean()),
                    float(input_tensor.std()),
                )
            )
            report_lines.append(f"  input_first_frame_first_12_values: {np.round(input_tensor[0].reshape(-1)[:12], 6).tolist()}")
            report_lines.append(f"  torch_score: {np.round(np.reshape(torch_score, [-1]), 6).tolist()}")
            report_lines.append(f"  torch_color_output: {np.round(np.reshape(torch_color, [-1, 3]), 6).tolist()}")
            report_lines.append(f"  torch_color_validation_pass: {bool(torch_flag)}")
            report_lines.append(f"  torch_color_validation_detail: {torch_pairs}")

            ort_color, ort_score = ort_worker.forward(input_tensor)
            ort_flag, ort_pairs = validate_color(ort_color, sample["sc_flag"])
            score_diff = np.max(np.abs(np.reshape(torch_score, [-1]) - np.reshape(ort_score, [-1])))
            color_diff = np.max(np.abs(np.reshape(torch_color, [-1]) - np.reshape(ort_color, [-1])))
            report_lines.append(f"  onnx_score: {np.round(np.reshape(ort_score, [-1]), 6).tolist()}")
            report_lines.append(f"  onnx_color_output: {np.round(np.reshape(ort_color, [-1, 3]), 6).tolist()}")
            report_lines.append(f"  onnx_color_validation_pass: {bool(ort_flag)}")
            report_lines.append(f"  onnx_color_validation_detail: {ort_pairs}")
            report_lines.append(f"  max_abs_diff_score: {float(score_diff):.8f}")
            report_lines.append(f"  max_abs_diff_color: {float(color_diff):.8f}")

        report_lines.append("")

    write_report(args.report_txt, report_lines)
    print(f"report_written: {args.report_txt}")


if __name__ == "__main__":
    main()
