import torch.onnx
import torch
import networks
from networks import load_checkpoint
import os
import argparse

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def parse_args():
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--data_path', type=str, default="../data/sample", help='数据文件地址')
    parser.add_argument('--model_path', type=str, default="../resources", help='模型位置')
    parser.add_argument('--img_size', type=int, default=256, help="height")
    parser.add_argument('--network', type=str, default='MoEA', help='网络解构分类')
    parser.add_argument('--model', type=str, default='', help='网络下的子模型')
    parser.add_argument('--threshold', type=float, default=-1, help='threshold')
    parser.add_argument("--infer_type", type=str, default="sdepth", help="推理类型")
    parser.add_argument("--min_color_check", type=int, default=4, help="颜色验证下限")
    parser.add_argument(
        '--checkpoint_path',
        type=str,
        default='',
        help='显式指定待导出的 checkpoint；为空时默认使用 <model_path>/<model>/model_best.pth.tar',
    )
    parser.add_argument(
        '--export_file',
        type=str,
        default='',
        help='显式指定导出的 onnx 文件路径；为空时默认输出到 <model_path>/<model>_<infer_type>.onnx',
    )
    args = parser.parse_args()
    if args.model == '':
        args.model = args.network
    return args


args = parse_args()

network = getattr(networks, args.network)

if __name__ == '__main__':
    with torch.no_grad():
        model = getattr(network, args.model)(**{"infer_type": args.infer_type})
        checkpoint_path = args.checkpoint_path or os.path.join(args.model_path, args.model, "model_best.pth.tar")
        load_checkpoint(checkpoint_path, model)
        model.eval()
        model = model.to(device)

        export_file = args.export_file or os.path.join(args.model_path, "%s_%s.onnx" % (args.model, args.infer_type))

        dummy_input = torch.randn(6, 3, 256, 256)
        if USE_CUDA:
            dummy_input = dummy_input.cuda()
        if args.infer_type == "score":
            torch.onnx.export(model, dummy_input, export_file, export_params=True, verbose=False, input_names=['input'],
                              output_names=["color", "score"], keep_initializers_as_inputs=True, opset_version=11)
        else:
            torch.onnx.export(model, dummy_input, export_file, export_params=True, verbose=False, input_names=['input'],
                              output_names=["map"], keep_initializers_as_inputs=True, opset_version=11)
