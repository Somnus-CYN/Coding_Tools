"""
Export Ultralytics YOLO11n to ONNX with a fixed input resolution.
This script is intended to be run on desktop during the packaging step.
"""
import argparse
import pathlib
import yaml
from ultralytics import YOLO


def load_config(cfg_path: pathlib.Path) -> dict:
    with cfg_path.open("r") as f:
        return yaml.safe_load(f)


def parse_args():
    parser = argparse.ArgumentParser(description="Export YOLO11n model to ONNX")
    parser.add_argument("--config", type=pathlib.Path, default=pathlib.Path("config.yaml"), help="Path to config.yaml")
    parser.add_argument("--opset", type=int, default=12, help="ONNX opset version")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    input_size = int(cfg.get("input_size", 640))
    paths = cfg["paths"]
    model_name = cfg.get("model_name", "yolo11n")

    weights_path = pathlib.Path(paths.get("weights", "./models/yolo11n.pt"))
    onnx_out = pathlib.Path(paths.get("onnx", "./models/yolo11n.onnx"))
    onnx_out.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading Ultralytics model {model_name} from {weights_path}…")
    model = YOLO(model_name if weights_path.name == f"{model_name}.pt" else weights_path)

    print(f"Exporting to ONNX at {onnx_out} with input size {input_size}x{input_size}, opset {args.opset}")
    # Use Ultralytics built-in export to keep graph simple and exclude NMS.
    model.export(
        format="onnx",
        imgsz=input_size,
        opset=args.opset,
        simplify=True,
        nms=False,  # keep postprocess outside the graph
        dynamic=False,
        half=False,
        device="cpu",
        outfile=str(onnx_out),
    )
    print("Export complete. Verify the file exists:", onnx_out.exists())


if __name__ == "__main__":
    main()
