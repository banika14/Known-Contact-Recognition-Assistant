import time
import cv2
import face_recognition
import numpy as np
from ultralytics import YOLO

# ============================== CONFIG =====================================
VIDEO_PATH = "test_video.mp4"
OUT_PATH = "known_contacts_demo.mp4"

KNOWN_CONTACTS = {
    "XYZ": {
        "image_file": "person-1.PNG",
        "note": "Medical CV Researcher"
    },
    "ABC": {
        "image_file": "person-2.PNG",
        "note": "Undergrad Thesis Mentee"
    }
}

TARGET_CLASSES = {0: "PERSON", 15: "CAT", 16: "DOG"}
CONF = 0.45
DANGER_ZONE_Y_RATIO = 0.75
SOCIAL_TRIGGER_DIST_M = 2.5
FACE_MATCH_TOLERANCE = 0.6
FACE_CHECK_EVERY_N_FRAMES = 5
CROP_PAD_PX = 20

PANEL_W, PANEL_H = 300, 70
PERSON_COLOR = (0, 255, 0)
ANIMAL_COLOR = (0, 255, 255) # High-contrast yellow
CONTACT_COLOR = (255, 200, 0)


# ============================ COLLISION HELPERS ==============================
def is_overlap(r1, r2, padding=10):
    return not (r1[2] + padding < r2[0] or
                r1[0] - padding > r2[2] or
                r1[3] + padding < r2[1] or
                r1[1] - padding > r2[3])

def get_adaptive_panel_position(target_box, occupied_rects, frame_w, frame_h):
    tx1, ty1, tx2, ty2 = target_box
    margin = 15
    candidates = [
        (tx2 + margin, ty1),
        (tx1 - PANEL_W - margin, ty1),
        (tx1, ty2 + margin),
        (tx1, ty1 - PANEL_H - margin),
        (tx2 + margin, ty2 - PANEL_H),
        (tx1 - PANEL_W - margin, ty2 - PANEL_H)
    ]

    for cx, cy in candidates:
        if cx < 0 or cy < 0 or cx + PANEL_W > frame_w or cy + PANEL_H > frame_h:
            continue
        panel_rect = [cx, cy, cx + PANEL_W, cy + PANEL_H]
        if not any(is_overlap(panel_rect, rect) for rect in occupied_rects):
            return cx, cy, panel_rect

    fallback_x = max(0, min(tx1, frame_w - PANEL_W))
    fallback_y = min(frame_h - PANEL_H, ty2 + margin)
    return fallback_x, fallback_y, [fallback_x, fallback_y, fallback_x + PANEL_W, fallback_y + PANEL_H]


# ============================ LOAD CONTACTS =================================
def load_known_contacts(contacts: dict):
    encodings, names, notes = [], [], []
    for name, data in contacts.items():
        try:
            img = face_recognition.load_image_file(data["image_file"])
            enc = face_recognition.face_encodings(img)
            if not enc:
                print(f"Warning: no face found for {name}, skipping.")
                continue
            encodings.append(enc[0])
            names.append(name)
            notes.append(data.get("note", ""))
        except Exception as e:
            print(f"Warning: could not load photo for {name}: {e}")
    return encodings, names, notes


# ============================ RENDER HELPERS =================================
def draw_box_with_label(frame, box, color, text, thickness=2, font_scale=0.6, font_thickness=2):
    x1, y1, x2, y2 = box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    # Black outline drop-shadow for high legibility
    cv2.putText(frame, text, (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thickness + 2)
    # Colored text
    cv2.putText(frame, text, (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, font_thickness)


def draw_contact_panel(frame, xy, name, note):
    x, y = xy
    cv2.rectangle(frame, (x, y), (x + PANEL_W, y + PANEL_H), (0, 0, 0), cv2.FILLED)
    cv2.rectangle(frame, (x, y), (x + PANEL_W, y + PANEL_H), CONTACT_COLOR, 2)
    cv2.putText(frame, name, (x + 10, y + 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.line(frame, (x + 10, y + 32), (x + PANEL_W - 10, y + 32), CONTACT_COLOR, 1)
    cv2.putText(frame, note, (x + 10, y + 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (210, 210, 210), 1)


# ================================== MAIN ====================================
def main():
    known_encodings, known_names, known_notes = load_known_contacts(KNOWN_CONTACTS)
    print(f"Loaded {len(known_names)} enrolled contact(s).")

    model = YOLO("yolov8n.pt")
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise ValueError(f"Cannot open {VIDEO_PATH}")

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    out = cv2.VideoWriter(OUT_PATH, cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_w, frame_h))

    danger_y = int(frame_h * DANGER_ZONE_Y_RATIO)
    last_match_cache = {}
    frame_idx = 0
    t0 = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        results = model(frame, classes=list(TARGET_CLASSES), conf=CONF, verbose=False)[0]

        # Gather all occupied bounding boxes for collision avoidance
        occupied_rects = []
        for box in results.boxes:
            bx1, by1, bx2, by2 = map(int, box.xyxy[0].tolist())
            occupied_rects.append([bx1, by1, bx2, by2])

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cls_id = int(box.cls[0].item())
            class_name = TARGET_CLASSES.get(cls_id, "OBJECT")
            dist_m = max(0.1, round(1.5 * (frame_h - y2) / (frame_h - danger_y), 1))

            if cls_id != 0:
                draw_box_with_label(frame, (x1, y1, x2, y2), ANIMAL_COLOR,
                                     f"{class_name} | {dist_m}m", font_scale=0.85, font_thickness=2)
                continue

            draw_box_with_label(frame, (x1, y1, x2, y2), PERSON_COLOR, f"Person | {dist_m}m")

            if dist_m > SOCIAL_TRIGGER_DIST_M or not known_encodings:
                continue

            key = (x1 // 40, y1 // 40)
            cached = last_match_cache.get(key)
            run_face_check = (cached is None) or (frame_idx % FACE_CHECK_EVERY_N_FRAMES == 0)

            # Properly initialize name and note variables
            name, note = (cached[0], cached[1]) if cached else (None, None)

            if run_face_check:
                cx1, cy1 = max(0, x1 - CROP_PAD_PX), max(0, y1 - CROP_PAD_PX)
                cx2, cy2 = min(frame_w, x2 + CROP_PAD_PX), min(frame_h, y2 + CROP_PAD_PX)
                crop = frame[cy1:cy2, cx1:cx2]
                rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                face_locs = face_recognition.face_locations(rgb)

                name, note = None, None
                if face_locs:
                    face_encs = face_recognition.face_encodings(rgb, [face_locs[0]])
                    matches = face_recognition.compare_faces(
                        known_encodings, face_encs[0], tolerance=FACE_MATCH_TOLERANCE)
                    if True in matches:
                        idx = matches.index(True)
                        name, note = known_names[idx], known_notes[idx]
                last_match_cache[key] = (name, note)

            if name:
                # Find an empty spot on the screen
                px, py, new_panel_rect = get_adaptive_panel_position((x1, y1, x2, y2), occupied_rects, frame_w, frame_h)
                occupied_rects.append(new_panel_rect)

                # Draw the panel and tether line
                draw_contact_panel(frame, (px, py), name, note)

                center_x, center_y = x1 + (x2 - x1)//2, y1 + (y2 - y1)//2
                panel_anchor_x = px if px > x2 else px + PANEL_W
                panel_anchor_y = py + (PANEL_H // 2)
                cv2.line(frame, (center_x, y1), (panel_anchor_x, panel_anchor_y), CONTACT_COLOR, 1)

        out.write(frame)

    cap.release()
    out.release()
    elapsed = time.time() - t0
    print(f"Done: {frame_idx} frames in {elapsed:.1f}s. Saved '{OUT_PATH}'.")

if __name__ == "__main__":
    main()
