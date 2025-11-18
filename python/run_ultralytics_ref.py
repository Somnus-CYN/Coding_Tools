"""
Run Ultralytics YOLO11n on test images and save detection results.
This serves as a reference baseline before export.
"""
import argparse
import json
import pathlib
from typing import List

import cv2
import yaml
from ultralytics import YOLO


def load_config(cfg_path: pathlib.Path) -> dict:
    with cfg_path.open("r") as f:
        return yaml.safe_load(f)


def serialize_detections(path: pathlib.Path, detections: List[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(detections, f, indent=2)
    print(f"Saved {len(detections)} image results to {path}")


def run_inference(model: YOLO, img_path: pathlib.Path, conf_thres: float):
    img = cv2.imread(str(img_path))
    results = model.predict(img, imgsz=model.model.args['imgsz'] if hasattr(model, 'model') else None, conf=conf_thres, verbose=False)
    outputs = []
    for r in results:
        for box in r.boxes:
            outputs.append(
                {
                    "bbox": box.xyxy[0].tolist(),
                    "score": float(box.conf[0]),
                    "class_id": int(box.cls[0]),
                    "class_name": r.names[int(box.cls[0])],
                }
            )
    return outputs


def main():
    parser = argparse.ArgumentParser(description="Run Ultralytics YOLO11n reference inference")
    parser.add_argument("--config", type=pathlib.Path, default=pathlib.Path("config.yaml"))
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths = cfg["paths"]
    conf_thres = float(cfg.get("confidence_threshold", 0.25))
    input_size = int(cfg.get("input_size", 640))

    model = YOLO(cfg.get("model_name", "yolo11n"))
    model.model.args['imgsz'] = input_size  # ensure consistency
    test_dir = pathlib.Path(paths.get("test_images", "../test_data/images"))

    detections = []
    for img_path in sorted(test_dir.glob("*")):
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
            continue
        outputs = run_inference(model, img_path, conf_thres)
        detections.append({"image": img_path.name, "detections": outputs})
        print(f"Processed {img_path.name} with {len(outputs)} detections")

    serialize_detections(pathlib.Path(paths.get("ref_ultralytics", "../test_data/ref_ultralytics.json")), detections)


if __name__ == "__main__":
    main()
