"""
Run ONNX Runtime inference for exported YOLO11n and save detections.
This script mirrors the preprocessing/postprocessing expected by NCNN backend.
"""
import argparse
import json
import pathlib
from typing import List, Tuple

import cv2
import numpy as np
import onnxruntime as ort
import yaml


def load_config(cfg_path: pathlib.Path) -> dict:
    with cfg_path.open("r") as f:
        return yaml.safe_load(f)


def letterbox(im: np.ndarray, new_shape: Tuple[int, int], color=(114, 114, 114)) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """Resize image with unchanged aspect ratio using padding."""
    shape = im.shape[:2]  # current shape [height, width]
    ratio = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * ratio)), int(round(shape[0] * ratio)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, ratio, (dw, dh)


def preprocess(img: np.ndarray, input_size: int):
    resized, ratio, (dw, dh) = letterbox(img, (input_size, input_size))
    resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    resized = resized.astype(np.float32) / 255.0
    resized = np.transpose(resized, (2, 0, 1))[None, ...]
    return resized, ratio, dw, dh


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def postprocess(outputs: List[np.ndarray], img_shape: Tuple[int, int], ratio: float, dw: float, dh: float, conf_thres: float, iou_thres: float, num_classes: int):
    # YOLO11n ONNX export yields a single output of shape [1, N, 85]
    preds = outputs[0]
    boxes = preds[..., :4]
    obj_conf = sigmoid(preds[..., 4:5])
    cls_conf = sigmoid(preds[..., 5:])
    scores = obj_conf * cls_conf

    # Convert boxes from center to corner format
    cx, cy, w, h = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2

    candidates = []
    for i in range(preds.shape[1]):
        class_id = int(np.argmax(scores[0, i]))
        conf = scores[0, i, class_id]
        if conf < conf_thres:
            continue
        bx1 = (x1[0, i] - dw) / ratio
        by1 = (y1[0, i] - dh) / ratio
        bx2 = (x2[0, i] - dw) / ratio
        by2 = (y2[0, i] - dh) / ratio
        candidates.append([bx1, by1, bx2, by2, float(conf), class_id])

    return nms(np.array(candidates), iou_thres)


def nms(boxes: np.ndarray, iou_thres: float):
    if boxes.size == 0:
        return []
    x1, y1, x2, y2, scores, cls = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3], boxes[:, 4], boxes[:, 5]
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(boxes[i])
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / ( (x2[i]-x1[i]) * (y2[i]-y1[i]) + (x2[order[1:]]-x1[order[1:]]) * (y2[order[1:]]-y1[order[1:]]) - inter )
        inds = np.where(iou <= iou_thres)[0]
        order = order[inds + 1]
    return [{"bbox": [float(k[0]), float(k[1]), float(k[2]), float(k[3])], "score": float(k[4]), "class_id": int(k[5])} for k in keep]


def run_session(session: ort.InferenceSession, img_path: pathlib.Path, input_size: int, conf_thres: float, iou_thres: float, num_classes: int):
    img = cv2.imread(str(img_path))
    blob, ratio, dw, dh = preprocess(img, input_size)
    inputs = {session.get_inputs()[0].name: blob}
    outputs = session.run(None, inputs)
    results = postprocess(outputs, img.shape[:2], ratio, dw, dh, conf_thres, iou_thres, num_classes)
    return results


def serialize(results: List[dict], path: pathlib.Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {len(results)} image results to {path}")


def main():
    parser = argparse.ArgumentParser(description="Run ONNX Runtime reference inference")
    parser.add_argument("--config", type=pathlib.Path, default=pathlib.Path("config.yaml"))
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths = cfg["paths"]
    input_size = int(cfg.get("input_size", 640))
    conf_thres = float(cfg.get("confidence_threshold", 0.25))
    iou_thres = float(cfg.get("nms_iou_threshold", 0.45))
    num_classes = len(cfg.get("classes", []))

    session = ort.InferenceSession(paths.get("onnx", "./models/yolo11n.onnx"), providers=["CPUExecutionProvider"])
    test_dir = pathlib.Path(paths.get("test_images", "../test_data/images"))

    results = []
    for img_path in sorted(test_dir.glob("*")):
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
            continue
        dets = run_session(session, img_path, input_size, conf_thres, iou_thres, num_classes)
        results.append({"image": img_path.name, "detections": dets})
        print(f"Processed {img_path.name} with {len(dets)} detections")

    serialize(results, pathlib.Path(paths.get("ref_onnx", "../test_data/ref_onnx.json")))


if __name__ == "__main__":
    main()
