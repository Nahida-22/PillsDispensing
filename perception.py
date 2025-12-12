# perception.py
#!/usr/bin/env python3
# coding=utf-8

import time

import cv2
from ultralytics import YOLO
import face_recognition

from patient_db import patient_db, normalize_name, label_to_medication
from arm_control import (
    arm_move,
    gripper_open,
    pick_pill_simple,
    P_PILL_1, P_PILL_2, P_PILL_3,
    P_PILL_4, P_PILL_5, P_PILL_6,
    P_PICK_1, P_PICK_2, P_PICK_3,
    P_PICK_4, P_PICK_5, P_PICK_6,
    P_CENTER_PLACE, P_CENTER_PLACE_2,
    P_ABOVE_PILL
)
# SHARED FRAME BUFFER
latest_frame = None

def set_latest_frame(frame):
    """Called by GUI to push the newest camera frame into this module."""
    global latest_frame
    latest_frame = frame

def get_latest_frame():
    global latest_frame
    if latest_frame is None:
        return None
    return latest_frame.copy()

# YOLO INIT
yolo_model = YOLO("/home/pi/Downloads/best (4).pt")

def yolo_detection(img):
    results = yolo_model(img, verbose=False)
    detections = []

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            label = yolo_model.names[cls]
            detections.append((label, conf, (x1, y1, x2, y2)))

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                img,
                f"{label} {conf:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )
    return img, detections

# FACE DETECTION
def detect_patient(img):
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    face_locs = face_recognition.face_locations(rgb)
    face_encs = face_recognition.face_encodings(rgb, face_locs)

    recognized_patients = []

    for idx, (top, right, bottom, left) in enumerate(face_locs):
        if idx >= len(face_encs):
            continue
        face_encoding = face_encs[idx]

        matches = face_recognition.compare_faces(
            patient_db.face_encodings,
            face_encoding,
            tolerance=0.5
        )
        name = "Unknown"
        patient_id = None
        patient_data = None

        if True in matches:
            first_match_index = matches.index(True)
            patient_id = patient_db.patient_names[first_match_index]
            patient_data = patient_db.patients.get(patient_id)
            if patient_data:
                name = patient_data["full_name"]

        color = (0, 255, 0) if patient_id else (0, 0, 255)
        cv2.rectangle(img, (left, top), (right, bottom), color, 2)

        info_line1 = f"Patient: {name}"
        cv2.putText(
            img,
            info_line1,
            (left, top - 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )

        if patient_data:
            info_line2 = f"Condition: {patient_data['condition']}"
            cv2.putText(
                img,
                info_line2,
                (left, top - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )

            recognized_patients.append({
                "id": patient_id,
                "name": name,
                "condition": patient_data["condition"],
                "medications": patient_data["medications"],
                "schedule": patient_data.get("schedule", {})
            })

    return img, recognized_patients


# CAPTURE FUNCTIONS
def capture_clear_image(position_name):
    print(f"[CAMERA] Capturing frame at {position_name}...")
    time.sleep(1.0)

    frame = get_latest_frame()
    if frame is not None:
        frame = cv2.resize(frame, (640, 480))
        print("[CAMERA] Frame captured")
        return frame
    else:
        print("[CAMERA] No frame available (camera not ready?)")
        return None


# SCAN + IMMEDIATE PICK FOR SLOT
def scan_and_pick_for_slot(patient, slot_key):
    """
    For a given patient and time-slot:
      - Go tray by tray
      - Run YOLO at each tray
      - If this tray has a needed med for this slot (and not already picked),
        immediately pick it and drop at center.
    Returns: (successful_picks, tray_moves_for_return)
    """
    patient_name = patient["name"]
    schedule = patient.get("schedule", {})
    slot_meds = schedule.get(slot_key, patient["medications"])

    if not slot_meds:
        print(f"\n[PICK] For {patient_name} at {slot_key}, no meds are scheduled.")
        return 0, []

    needed_meds_norm = {normalize_name(m): m for m in slot_meds}
    needed_meds_display = list(needed_meds_norm.values())

    print("\n" + "=" * 60)
    print(f"SCAN + IMMEDIATE PICK FOR {patient_name.upper()} – slot {slot_key}")
    print(f"Medications required: {', '.join(needed_meds_display)}")
    print("=" * 60)

    pill_positions = [
        ("Right_1", P_PILL_1, P_PICK_1),
        ("Right_2", P_PILL_2, P_PICK_2),
        ("Right_3", P_PILL_3, P_PICK_3),
        ("Left_1",  P_PILL_4, P_PICK_4),
        ("Left_2",  P_PILL_5, P_PICK_5),
        ("Left_3",  P_PILL_6, P_PICK_6),
    ]

    gripper_open()
    arm_move(P_ABOVE_PILL[:5], 1000)

    picked_meds_norm = set()  # which meds (normalized) we already picked
    tray_moves = []           # to later return trays
    successful_picks = 0

    for position_name, pill_scan_pos, pill_pick_pos in pill_positions:
        if len(picked_meds_norm) == len(needed_meds_norm):
            print("[PICK] All required meds already picked for this slot.")
            break

        print(f"\n[SCAN] Moving to {position_name} for detection")
        arm_move(pill_scan_pos[:5], 1000)
        time.sleep(0.5)

        frame = capture_clear_image(position_name)
        if frame is None:
            print("[SCAN] No frame captured, skipping this position.")
            arm_move(P_ABOVE_PILL[:5], 1000)
            continue

        detected_img, detections = yolo_detection(frame.copy())

        if not detections:
            print("[SCAN] No pills detected at this tray.")
            arm_move(P_ABOVE_PILL[:5], 1000)
            continue

        print(f"[SCAN] Found {len(detections)} object(s) at {position_name}:")
        meds_here = set()
        for (label, conf, bbox) in detections:
            med_name_raw = label_to_medication(label)
            med_key = normalize_name(med_name_raw)
            print(f"   • {label} ({conf:.2f}) -> mapped to '{med_name_raw}'")
            if med_key in needed_meds_norm:
                meds_here.add(needed_meds_norm[med_key])

        if not meds_here:
            print("[SCAN] Tray meds do not match this patient's schedule.")
            arm_move(P_ABOVE_PILL[:5], 1000)
            continue

        # Choose one med from this tray that we still need
        med_to_pick = None
        for m in slot_meds:  # keep original order
            if m in meds_here and normalize_name(m) not in picked_meds_norm:
                med_to_pick = m
                break

        if med_to_pick is None:
            print("[SCAN] All meds on this tray already picked previously.")
            arm_move(P_ABOVE_PILL[:5], 1000)
            continue

        med_index = len(picked_meds_norm)
        if med_index == 0:
            drop_position = P_CENTER_PLACE
            drop_name = "P_CENTER_PLACE"
        else:
            drop_position = P_CENTER_PLACE_2
            drop_name = "P_CENTER_PLACE_2"

        print(f"[PICK] Tray {position_name} has required med: {med_to_pick}")
        print(f"[PICK] Immediately picking this tray and dropping at {drop_name}")

        success = pick_pill_simple(position_name, pill_pick_pos, drop_position)

        if success:
            successful_picks += 1
            picked_meds_norm.add(normalize_name(med_to_pick))
            tray_moves.append({
                "position": position_name,
                "pick_pos": pill_pick_pos,
                "drop_pos": drop_position
            })
            print(f"[PICK] Completed pick for medication: {med_to_pick}")
        else:
            print(f"[PICK] Failed to pick tray {position_name} for med {med_to_pick}")

        arm_move(P_ABOVE_PILL[:5], 1000)
        time.sleep(0.3)

    print(f"\n[PICK] SUMMARY for {patient_name} at {slot_key}: "
          f"{successful_picks} tray(s) picked for {len(slot_meds)} medication(s).")
    return successful_picks, tray_moves