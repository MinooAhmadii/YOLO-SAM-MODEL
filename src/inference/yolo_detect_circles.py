from pathlib import Path
from typing import Dict, Any, List

import cv2
import numpy as np
from ultralytics import YOLO


def _load_model(weights_path: str) -> YOLO:
    # Lazily load YOLO from the given path
    return YOLO(weights_path)


def detect_and_draw(model: YOLO, image_path: str, save_path: str) -> float | None:
    """Run YOLO, draw simple disc/cup circles, write image; return CDR (or None)."""
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not read {image_path}")
        return None

    results = model(image_path, conf=0.25)

    disc_radius = 0
    cup_radius = 0

    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cls = int(box.cls[0])

            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            radius = int(((x2 - x1) + (y2 - y1)) / 4)

            if cls == 0:  # Disc
                cv2.circle(img, (cx, cy), radius, (0, 255, 0), 3)
                disc_radius = radius
            else:  # Cup
                cv2.circle(img, (cx, cy), radius, (0, 0, 255), 3)
                cup_radius = radius

    cdr = None
    if disc_radius > 0 and cup_radius > 0:
        cdr = cup_radius / disc_radius
        if cdr < 0.5:
            status, color = "NORMAL", (0, 255, 0)
        elif cdr < 0.7:
            status, color = "SUSPECT", (0, 165, 255)
        else:
            status, color = "HIGH RISK", (0, 0, 255)
        cv2.putText(img, f"CDR: {cdr:.3f} - {status}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(save_path, img)
    print(f"Saved: {save_path}")
    return cdr


def main(params: Dict[str, Any], output_dir: Path) -> List[Path]:
    """
    Run circle-based YOLO overlay on all images in params['images_dir'].
    Returns the list of produced output files so DerivaML can track them.
    Expected params: weights_path, images_dir (and optionally output_dir).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    weights = params.get("weights_path", "models/best.pt")
    images_dir = Path(params.get("images_dir", "test_images"))

    model = _load_model(weights)

    produced: List[Path] = []
    for img in sorted(list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))):
        out = output_dir / f"circles_{img.stem}.jpg"
        detect_and_draw(model, str(img), str(out))
        produced.append(out)

    return produced


if __name__ == "__main__":
    params = {
        "images_dir": "test_images",
        "weights_path": "models/best.pt",
        "output_dir": "results"
    }
    _ = main(params, Path(params["output_dir"]))

