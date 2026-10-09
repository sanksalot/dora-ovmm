"""Client for the SAM3 detection server. See docker/sam3/app.py for the query/reply protocol."""

import json
import os
import re
import time
from pathlib import Path

import cv2
import numpy as np
from core.utils.zenoh_rpc import query, parameter

# Every call is saved here, one folder per run: what was asked, what came back, drawn
# on the frame. DETECTIONS=0 switches it off.
DETECTIONS = Path(__file__).resolve().parents[2] / "outputs/detections"


class ObjectNotFound(RuntimeError):
    """The detector replied successfully but found no matching instance."""


def _slug(text):
    return re.sub("[^a-z0-9]+", "-", str(text).lower()).strip("-") or "unnamed"


def run_folder():
    """This run's folder. core.pipeline.mission_tree and web.server name the run; a tool
    started by hand gets a name of its own, kept in the environment so that a pick it
    starts saves beside it."""
    run = os.environ.setdefault("MISSION_RUN_ID", time.strftime("%Y%m%d-%H%M%S") + "_manual")
    return DETECTIONS / run


BEST, OTHER = (80, 255, 80), (0, 200, 255)  # BGR: the chosen instance, the rest
TINT = 0.6  # how much of the mask's colour shows through


def tag(picture, text, at, colour, scale=0.55):
    """A filled label at pixel `at` (its bottom-left), in `colour` with dark text, kept on
    the image. For a mask or a grasp, so its name and score read on any background."""
    (width, height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    x = int(np.clip(at[0], 0, picture.shape[1] - width - 8))
    y = int(np.clip(at[1], height + 8, picture.shape[0]))
    cv2.rectangle(picture, (x, y - height - 8), (x + width + 8, y), colour, -1)
    cv2.putText(picture, text, (x + 4, y - 4), cv2.FONT_HERSHEY_SIMPLEX, scale, (20, 20, 20), 1, cv2.LINE_AA)


def draw_mask(picture, mask, colour, label=None, thick=True):
    """Tint the mask in `colour`, outline it with a dark halo so it reads on any
    background, and tag it above its top with `label`."""
    picture[mask] = ((1 - TINT) * picture[mask] + TINT * np.array(colour)).astype(np.uint8)
    outline, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(picture, outline, -1, (20, 20, 20), 4 if thick else 3, cv2.LINE_AA)
    cv2.drawContours(picture, outline, -1, colour, 2 if thick else 1, cv2.LINE_AA)
    if label and mask.any():
        x, y, _, _ = cv2.boundingRect(mask.astype(np.uint8))
        tag(picture, label, (x, y - 4), colour)


def save_detection(image_bgr, prompt, masks, boxes, scores):
    """The frame with every mask on it, the best one brightest, and a row in index.jsonl.

    For showing afterwards what the detector was asked and saw, so a full disk or a
    bad frame is reported and never becomes a failed detection."""
    if os.environ.get("DETECTIONS", "1") == "0":
        return
    try:
        folder = run_folder()
        folder.mkdir(parents=True, exist_ok=True)
        stage = os.environ.get("MISSION_STAGE", "manual")
        best = int(np.argmax(scores)) if len(scores) else None
        picture = image_bgr.copy()
        for index in sorted(range(len(scores)), key=lambda i: i == best):  # the best drawn last, on top
            draw_mask(picture, masks[index], BEST if index == best else OTHER,
                      f"{prompt}  {scores[index]:.2f}" if index == best else f"{scores[index]:.2f}",
                      thick=index == best)
        found = f"{scores[best]:.2f}" if best is not None else "nothing found"
        caption = f'"{prompt}"  {found}  |  {stage}  |  {folder.name}'
        bar = np.full((26, picture.shape[1], 3), 30, dtype=np.uint8)
        cv2.putText(bar, caption, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        number = len(list(folder.glob("[0-9][0-9][0-9]_*.jpg"))) + 1
        score = f"{scores[best]:.2f}" if best is not None else "none"
        name = f"{number:03d}_{_slug(stage)}_{_slug(prompt)}_{score}.jpg"
        cv2.imwrite(str(folder / name), np.vstack([bar, picture]))
        with open(folder / "index.jsonl", "a") as index:
            index.write(json.dumps({
                "time": time.strftime("%Y-%m-%d %H:%M:%S"), "run": folder.name, "stage": stage,
                "query": prompt, "found": best is not None, "scores": [float(s) for s in scores],
                "boxes": np.asarray(boxes, dtype=float).round(1).tolist(), "file": name}) + "\n")
    except (OSError, cv2.error) as error:
        print(f"[SAM3] detection not saved: {error}", flush=True)


def detect_all(image_bgr, prompt, conf=0.5, timeout=30, *, metadata=None):
    """Return masks, xyxy boxes, scores, and echoed observation metadata.

    Empty results are valid. Indices are local to this image, not tracking IDs.
    Caller-supplied object_id identifies the requested target, not every mask.
    """
    # Compress the image before sending it to the detector.
    ok, jpeg = cv2.imencode(".jpg", image_bgr)
    if not ok:
        raise ValueError("Could not encode image")
    meta, body = query(
        f"sam3/detect?prompt={parameter(prompt)};conf={conf}", jpeg.tobytes(), timeout,
        metadata=metadata,
    )
    count, height, width = (meta["num_instances"], meta["height"], meta["width"])
    # Split the reply into masks, bounding boxes, and confidence scores.
    masks_end = count * height * width
    boxes_end = masks_end + count * 4 * 4
    # Check the exact reply size before interpreting its bytes as arrays.
    if len(body) != boxes_end + count * 4:
        raise RuntimeError("Malformed SAM3 reply")
    # Decode one mask, box, and score per detected object.
    masks = np.frombuffer(body[:masks_end], dtype=bool).reshape(count, height, width)
    boxes = np.frombuffer(body[masks_end:boxes_end], dtype=np.float32).reshape(count, 4).copy()
    scores = np.frombuffer(body[boxes_end:], dtype=np.float32)
    target_h, target_w = image_bgr.shape[:2]
    # Resize detections back to the original image size when needed.
    if (height, width) != (target_h, target_w):
        masks = np.asarray([
            cv2.resize(m.astype(np.uint8), (target_w, target_h), interpolation=cv2.INTER_NEAREST)
            for m in masks
        ], dtype=bool).reshape(count, target_h, target_w)
        boxes *= np.array([target_w / width, target_h / height] * 2)
    save_detection(image_bgr, prompt, masks, boxes, scores)
    return {"masks": masks, "boxes": boxes, "scores": scores, "metadata": meta}


def detect(image_bgr, prompt, conf=0.5, timeout=30, *, metadata=None):
    """Compatibility API: return the highest-scoring instance as (mask, score)."""
    result = detect_all(image_bgr, prompt, conf, timeout, metadata=metadata)
    if not len(result["scores"]):
        raise ObjectNotFound(f"SAM3 found no instance of '{prompt}'")
    # Choose the most confident instance when the caller wants only one object.
    best = int(np.argmax(result["scores"]))
    return result["masks"][best], float(result["scores"][best])
