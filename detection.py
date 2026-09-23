"""
Real-Time Individual Apple Counting using YOLO26 - line-crossing edition
---------------------------------------------------------------------------
Detects EACH apple individually (even in a dense, overlapping pile),
tracks it with a stable ID, colors it by CLASS, and only adds it to the
count the moment it crosses a reference line - the same idea as a real
conveyor-belt counting sensor, instead of counting everything visible.

USAGE:
    python detection.py --source videos/input.mp4 --weights models/best.pt

If you've calibrated a custom line (calibrate_counting_line.py), that's
used automatically. Otherwise a default horizontal line is drawn across
the frame at --line-y (fraction of frame height, default 0.55).
"""

import argparse
import os
import time
import cv2
from ultralytics import YOLO

from utils.tiling import tiled_predict
from utils.tracker import CentroidTracker
from utils.hud import draw_apple_box, draw_hud_panel, build_class_colors
from utils.drawing import draw_counting_line
from utils.config import load_calibration
from utils.crossing import side_of_line, default_line


def get_args():
    parser = argparse.ArgumentParser(description="Individual apple detection, tracking & line-crossing counting with YOLO26")
    parser.add_argument("--source", type=str, default="videos/input.mp4",
                         help="Path to input video file")
    parser.add_argument("--weights", type=str, default="yolo26n.pt",
                         help="Model weights - use models/best.pt for your custom-trained model")
    parser.add_argument("--conf", type=float, default=0.2,
                         help="Confidence threshold for detections (lower = catches more low-confidence/occluded apples, at the cost of more false positives)")
    parser.add_argument("--output", type=str, default="videos/output_counted.mp4",
                         help="Path to save the annotated output video")
    parser.add_argument("--imgsz", type=int, default=640,
                         help="Inference resolution per tile")
    parser.add_argument("--tile-size", type=int, default=640,
                         help="Tile size in pixels for splitting dense/crowded frames")
    parser.add_argument("--overlap", type=float, default=0.3,
                         help="Fractional overlap between tiles (helps catch apples split across tile edges)")
    parser.add_argument("--no-tile", action="store_true",
                         help="Disable tiling and run detection on the full frame only (faster, worse on dense piles)")
    parser.add_argument("--classes", type=str, default=None,
                         help="Comma-separated class names to detect (default: all classes in the model)")
    parser.add_argument("--line-y", type=float, default=0.55,
                         help="Fraction of frame height for the default counting line's center (ignored if a calibrated line exists)")
    parser.add_argument("--line-tilt", type=float, default=0.25,
                         help="How diagonal the default line is, as a fraction of frame height (0 = flat horizontal, higher = steeper)")
    parser.add_argument("--show-conf", action="store_true",
                         help="Show confidence score in each box label (off by default to reduce clutter)")
    parser.add_argument("--show", action="store_true",
                         help="Show a live preview window while processing")
    return parser.parse_args()


def resolve_class_ids(model, requested_names):
    name_to_id = {name.lower(): cid for cid, name in model.names.items()}
    if requested_names is None:
        return list(model.names.keys())
    ids = []
    for name in requested_names.split(","):
        name = name.strip().lower()
        if name in name_to_id:
            ids.append(name_to_id[name])
        else:
            print(f"Warning: class '{name}' not found in model, skipping. "
                  f"Available classes: {list(name_to_id.keys())}")
    return ids


def main():
    args = get_args()

    if not os.path.exists(args.source):
        raise FileNotFoundError(
            f"Could not find input video at '{args.source}'. "
            f"Put your video inside the 'videos' folder and update --source if needed."
        )

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    model = YOLO(args.weights)
    class_ids = resolve_class_ids(model, args.classes)
    if not class_ids:
        raise ValueError("No valid classes to detect. Check --classes against the model's class list.")

    class_colors = build_class_colors(model.names)

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video '{args.source}'")

    fps_in = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    counting_line = load_calibration()
    line_source = "calibrated"
    if not counting_line:
        counting_line = default_line(width, height, args.line_y, args.line_tilt)
        line_source = "default"
    print(f"Using {line_source} counting line: {counting_line}")

    writer = cv2.VideoWriter(
        args.output,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps_in,
        (width, height),
    )

    tracker = CentroidTracker(max_distance=55, max_missed_frames=15)
    count_history = []
    id_side = {}                                            # track_id -> last known side of the line
    counted_ids = set()                                      # track_ids already counted (crossed once)
    class_counts = {model.names[c]: 0 for c in class_ids}     # class name -> unique count

    frame_idx = 0
    t_start = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        t0 = time.time()

        if args.no_tile:
            results = model.predict(frame, classes=class_ids, conf=args.conf,
                                     imgsz=args.imgsz, verbose=False)
            boxes, confs, det_classes = [], [], []
            for r in results:
                if r.boxes is None:
                    continue
                xyxy = r.boxes.xyxy.cpu().numpy()
                confs_r = r.boxes.conf.cpu().numpy()
                cls_r = r.boxes.cls.cpu().numpy().astype(int)
                for (x1, y1, x2, y2), c, cid in zip(xyxy, confs_r, cls_r):
                    boxes.append([x1, y1, x2, y2])
                    confs.append(float(c))
                    det_classes.append(int(cid))
        else:
            boxes, confs, det_classes = tiled_predict(
                model, frame, class_ids,
                conf=args.conf, imgsz=args.imgsz,
                tile_size=args.tile_size, overlap=args.overlap,
            )

        ids = tracker.update(boxes, [model.names[c] for c in det_classes], confs)

        for track_id, (x1, y1, x2, y2), class_name, conf, _was_detected in ids:
            color = class_colors.get(class_name, (255, 255, 255))
            draw_apple_box(frame, int(x1), int(y1), int(x2), int(y2), track_id, conf,
                            class_name, color, show_conf=args.show_conf)

            # Line-crossing count: only counts the first time this ID
            # switches from one side of the line to the other. Using the
            # smoothed/coasted position here too, so a crossing isn't missed
            # just because that exact frame had a detection gap.
            centroid = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
            current_side = side_of_line(centroid, counting_line[0], counting_line[1])
            prev_side = id_side.get(track_id)

            if current_side != 0:
                if prev_side is not None and prev_side != current_side and track_id not in counted_ids:
                    counted_ids.add(track_id)
                    class_counts[class_name] = class_counts.get(class_name, 0) + 1
                id_side[track_id] = current_side

        draw_counting_line(frame, counting_line)

        avg_conf = sum(confs) / len(confs) if confs else 0.0
        total_count = len(counted_ids)
        count_history.append(total_count)
        elapsed = time.time() - t_start
        proc_fps = frame_idx / elapsed if elapsed > 0 else 0.0

        draw_hud_panel(
            frame,
            total_count=total_count,
            current_count=len(ids),
            history=count_history,
            class_counts=class_counts,
            class_colors=class_colors,
            avg_conf=avg_conf,
            fps=proc_fps,
        )

        writer.write(frame)

        if args.show:
            cv2.imshow("Apple Counter", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        if frame_idx % 30 == 0:
            breakdown = ", ".join(f"{k}: {v}" for k, v in class_counts.items())
            print(f"Frame {frame_idx} | Crossed so far: {total_count} ({breakdown}) | "
                  f"Tracked now: {len(ids)} | {time.time() - t0:.2f}s/frame")

    cap.release()
    writer.release()
    if args.show:
        cv2.destroyAllWindows()

    print(f"\nDone! Final counts (crossed the line): {class_counts}")
    print(f"Total unique apples counted: {len(counted_ids)}")
    print(f"Annotated video saved to: {args.output}")


if __name__ == "__main__":
    main()
