"""
animal_detect.py

Usage:
    python animal_detect.py              # uses pretrained yolov5s (COCO) - will detect 'elephant'
    python animal_detect.py --weights best.pt   # use custom YOLOv5 weights (for pig + elephant)
    python animal_detect.py --source 0   # webcam (default)
    python animal_detect.py --source video.mp4  # video file

Notes:
- If your model does not include 'pig' in its class names, pig won't be detected until you provide custom weights trained on pigs.
"""

import argparse
import time
import sys

import cv2
import torch
import numpy as np

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", type=str, default=None,
                   help="Path to custom YOLO .pt weights. If omitted, uses pretrained yolov5s.")
    p.add_argument("--source", type=str, default="0",
                   help="Video source (0 for webcam) or file path")
    p.add_argument("--conf", type=float, default=0.4, help="Confidence threshold")
    p.add_argument("--save", action="store_true", help="Save frames where target detected into ./detections/")
    return p.parse_args()

def main():
    args = parse_args()

    # Load model
    if args.weights:
        print(f"[INFO] Loading custom weights from {args.weights} ...")
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=args.weights, force_reload=False)
    else:
        print("[INFO] Loading pretrained yolov5s (COCO classes) ...")
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

    model.conf = args.conf  # confidence threshold

    # Class names dict
    names = model.names  # dict {class_index: class_name}
    print(f"[INFO] Model classes: {len(names)} classes loaded.")
    # Find indices for elephant and pig if present
    target_classes = {}
    for idx, nm in names.items():
        if nm.lower() == "elephant":
            target_classes["elephant"] = idx
        if nm.lower() == "pig":
            target_classes["pig"] = idx

    print("[INFO] Targets found in model:", target_classes)

    # If pig not found, warn user
    if "elephant" not in target_classes:
        print("[WARNING] 'elephant' class not found in this model. You may need a custom model trained for elephants.")
    if "pig" not in target_classes:
        print("[WARNING] 'pig' class not found in this model. To detect pigs, provide custom weights trained with pig labels.")

    # Open video source
    src = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print("[ERROR] Cannot open video source:", src)
        sys.exit(1)

    save_idx = 0
    if args.save:
        import os
        os.makedirs("detections", exist_ok=True)

    print("[INFO] Starting detection. Press 'q' to quit.")
    prev_time = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] Stream ended or frame not received.")
            break

        # YOLOv5 expects RGB
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Inference (returns a special results object)
        results = model(img, size=640)  # increase size for better accuracy if GPU available

        # pandas-like results
        df = results.pandas().xyxy[0]  # xmin, ymin, xmax, ymax, confidence, class, name

        detected_targets = []
        for _, row in df.iterrows():
            cls_name = str(row['name']).lower()
            conf = float(row['confidence'])
            if cls_name in ("elephant", "pig"):
                xmin, ymin, xmax, ymax = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
                label = f"{cls_name} {conf:.2f}"
                # Draw box
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                cv2.putText(frame, label, (xmin, max(ymin-10,0)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
                detected_targets.append((cls_name, conf, (xmin, ymin, xmax, ymax)))

        # Show FPS
        cur_time = time.time()
        fps = 1.0 / (cur_time - prev_time) if (cur_time - prev_time) > 0 else 0.0
        prev_time = cur_time
        cv2.putText(frame, f"FPS: {fps:.1f}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,255), 2)

        # Show frame
        cv2.imshow("Animal Detector", frame)

        # If detections, print and optionally save
        if detected_targets:
            for dt in detected_targets:
                name, conf, bbox = dt
                print(f"[DETECTED] {name} (conf {conf:.2f}) bbox={bbox}")
            if args.save:
                fname = f"detections/detect_{save_idx}.jpg"
                cv2.imwrite(fname, frame)
                print(f"[INFO] Saved {fname}")
                save_idx += 1

            # Optional: play a beep on detection (cross-platform limited)
            try:
                import winsound
                winsound.Beep(1000, 300)
            except Exception:
                # linux / mac simple beep via terminal
                sys.stdout.write('\a')
                sys.stdout.flush()

        # Quit key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Exited.")

if __name__ == "__main__":
    main()