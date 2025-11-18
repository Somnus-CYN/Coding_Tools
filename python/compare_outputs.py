"""
Sanity check comparing Ultralytics and ONNX Runtime detection outputs.
This is a loose comparison (not a full evaluation), focusing on bounding box closeness.
"""
import argparse
import json
import pathlib
from typing import Dict, List, Tuple

import numpy as np


def load_json(path: pathlib.Path) -> List[Dict]:
    with path.open("r") as f:
        return json.load(f)


def iou(box1: List[float], box2: List[float]) -> float:
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def match_detections(ref: List[Dict], cand: List[Dict], iou_thres: float = 0.5) -> Tuple[int, int]:
    matched = 0
    for r in ref:
        for c in cand:
            if r["class_id"] != c["class_id"]:
                continue
            if iou(r["bbox"], c["bbox"]) >= iou_thres:
                matched += 1
                break
    return matched, len(ref)


def summarize(ref_path: pathlib.Path, onnx_path: pathlib.Path, report_path: pathlib.Path):
    ref = load_json(ref_path)
    onnx = load_json(onnx_path)
    onnx_map = {item["image"]: item for item in onnx}

    summary = {}
    total_ref = 0
    total_match = 0

    for ref_item in ref:
        image_name = ref_item["image"]
        cand_item = onnx_map.get(image_name, {"detections": []})
        matched, count = match_detections(ref_item["detections"], cand_item.get("detections", []))
        total_ref += count
        total_match += matched
        summary[image_name] = {
            "ref_detections": count,
            "onnx_detections": len(cand_item.get("detections", [])),
            "matched": matched,
            "match_ratio": matched / count if count else 0.0,
        }

    summary["overall"] = {
        "ref_total": total_ref,
        "matched_total": total_match,
        "match_ratio": total_match / total_ref if total_ref else 0.0,
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w") as f:
        json.dump(summary, f, indent=2)
    print("Comparison summary:")
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Compare Ultralytics and ONNX Runtime outputs")
    parser.add_argument("--ref", type=pathlib.Path, default=pathlib.Path("../test_data/ref_ultralytics.json"))
    parser.add_argument("--onnx", type=pathlib.Path, default=pathlib.Path("../test_data/ref_onnx.json"))
    parser.add_argument("--report", type=pathlib.Path, default=pathlib.Path("../test_data/compare_report.json"))
    args = parser.parse_args()

    summarize(args.ref, args.onnx, args.report)


if __name__ == "__main__":
    main()
