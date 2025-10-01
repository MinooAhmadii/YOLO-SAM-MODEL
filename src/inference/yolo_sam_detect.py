import os
import sys
from pathlib import Path
from typing import Dict, Any, List

import cv2
import numpy as np
import torch
from ultralytics import YOLO

# SAM
try:
    from segment_anything import sam_model_registry, SamPredictor
except ImportError:
    print("Please install SAM: pip install git+https://github.com/facebookresearch/segment-anything.git")
    sys.exit(1)


class GlaucomaDetector:
    def __init__(self, yolo_path: str = "models/best.pt", sam_checkpoint: str = "models/sam_vit_h_4b8939.pth"):
        """
        Initialize the Glaucoma Detection System with YOLO and SAM.

        Args:
            yolo_path: Path to your trained YOLO model
            sam_checkpoint: Path to SAM checkpoint (download from Meta)
        """
        # ---- device selection for torch models
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        # ---- YOLO
        print("Loading YOLO model...")
        self.yolo_model = YOLO(yolo_path)

        # ---- SAM
        if not os.path.exists(sam_checkpoint):
            print(f"SAM checkpoint not found at {sam_checkpoint}")
            print("Download it from: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth")
            sys.exit(1)

        print("Loading SAM model...")
        sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint)
        sam.to(self.device)
        sam.eval()
        self.predictor = SamPredictor(sam)

    def detect_with_sam(self, image_path: str, output_path: str) -> Dict[str, Any]:
        """
        Detect optic disc and cup using YOLO + SAM for precise segmentation.

        Returns a dict with metrics; writes a visualization to output_path.
        """
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Could not read image {image_path}")
            return {"image": Path(image_path).name, "cdr": None, "status": "Failed"}

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.predictor.set_image(img_rgb)

        # YOLO detection
        results = self.yolo_model(image_path, conf=0.25, iou=0.45, device=self.device)

        disc_area = disc_diam = cup_area = cup_diam = 0.0

        for r in results:
            if r.boxes is None:
                continue

            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cls = int(box.cls[0])
                conf = float(box.conf[0])

                input_box = np.array([x1, y1, x2, y2])

                masks, scores, _ = self.predictor.predict(
                    box=input_box,
                    multimask_output=True
                )

                best_idx = int(np.argmax(scores))
                mask = masks[best_idx].astype(np.uint8) * 255

                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if not contours:
                    continue

                largest = max(contours, key=cv2.contourArea)

                if cls == 0:  # Optic Disc
                    cv2.drawContours(img, [largest], -1, (0, 255, 0), 2)
                    disc_area = float(cv2.contourArea(largest))
                    (_, _), radius = cv2.minEnclosingCircle(largest)
                    disc_diam = float(radius * 2)
                    cv2.putText(img, f"Disc: {conf:.2f}", (int(x1), max(15, int(y1 - 10))),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                else:        # Optic Cup
                    cv2.drawContours(img, [largest], -1, (0, 0, 255), 2)
                    cup_area = float(cv2.contourArea(largest))
                    (_, _), radius = cv2.minEnclosingCircle(largest)
                    cup_diam = float(radius * 2)
                    cv2.putText(img, f"Cup: {conf:.2f}", (int(x1), max(15, int(y1 - 10))),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # Metrics
        cdr_area = cdr_diam = cdr_final = None
        if disc_area > 0 and cup_area > 0 and disc_diam > 0 and cup_diam > 0:
            cdr_area = float(np.sqrt(cup_area / disc_area))
            cdr_diam = float(cup_diam / disc_diam)
            cdr_final = float((cdr_area + cdr_diam) / 2)

            if cdr_final < 0.5:
                status, color = "NORMAL", (0, 255, 0)
            elif cdr_final < 0.7:
                status, color = "SUSPECT", (0, 165, 255)
            else:
                status, color = "HIGH RISK", (0, 0, 255)

            cv2.putText(img, f"CDR (Area): {cdr_area:.3f}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(img, f"CDR (Diam): {cdr_diam:.3f}", (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(img, f"CDR (Final): {cdr_final:.3f} - {status}", (20, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
            rim_area = disc_area - cup_area
            rim_ratio = rim_area / disc_area
            cv2.putText(img, f"Rim/Disc Ratio: {rim_ratio:.3f}", (20, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        else:
            status = "Detection incomplete"
            cv2.putText(img, status, (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Save only (no imshow for headless)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(output_path, img)
        print(f"Saved: {output_path}")

        return {
            "image": Path(image_path).name,
            "cdr": cdr_final,
            "cdr_area": cdr_area,
            "cdr_diam": cdr_diam,
            "status": status
        }

    def batch_process(self, input_folder: str, output_folder: str) -> List[Dict[str, Any]]:
        input_path = Path(input_folder)
        output_path = Path(output_folder)
        output_path.mkdir(parents=True, exist_ok=True)

        results: List[Dict[str, Any]] = []
        for img_file in sorted(list(input_path.glob("*.jpg")) + list(input_path.glob("*.png"))):
            print(f"\nProcessing: {img_file.name}")
            out_file = output_path / f"result_{img_file.stem}.jpg"
            metrics = self.detect_with_sam(str(img_file), str(out_file))
            results.append(metrics)

        # Write summary safely
        with open(output_path / "summary.txt", "w") as f:
            for r in results:
                cdr_str = f"{r['cdr']:.3f}" if r["cdr"] is not None else "N/A"
                f.write(f"{r['image']}: CDR={cdr_str}, Status={r['status']}\n")

        return results


# ---- Manual test (kept for convenience; won’t run in DerivaML) ----
if __name__ == "__main__":
    det = GlaucomaDetector(
        yolo_path="models/best.pt",
        sam_checkpoint="sam_vit_h_4b8939.pth"
    )
    det.detect_with_sam("test_images/BEH-9.png", "results/sam_result.jpg")


# ---- Library-style entrypoint for DerivaML driver ----
def main(params: Dict[str, Any], output_dir: Path) -> List[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    det = GlaucomaDetector(
        yolo_path=params.get("weights_path", "models/best.pt"),
        sam_checkpoint=params.get("sam_checkpoint", "sam_vit_h_4b8939.pth"),
    )
    images_dir = Path(params.get("images_dir", "test_images"))
    results = det.batch_process(str(images_dir), str(output_dir))

    produced: List[Path] = []
    for r in results:
        produced.append(output_dir / f"result_{Path(r['image']).stem}.jpg")
    summ = output_dir / "summary.txt"
    if summ.exists():
        produced.append(summ)
    return produced
