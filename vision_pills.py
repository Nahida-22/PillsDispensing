#!/usr/bin/env python3
# coding=utf-8
import cv2
import time
from ultralytics import YOLO
import face_recognition

def normalize_name(s: str) -> str:
    return "".join(c.lower() for c in s if c.isalnum())

CLASS_TO_MED = {
    "Vildaril": "Vildaril",
    "Clopica": "Clopica",
    "Osteocare": "Osteocare",
    "Flucoxacilin": "Flucoxacilin",
    "Imdex": "Imdex",
    "Doliprane": "Doliprane",
}
CLASS_TO_MED_NORM = {normalize_name(k): v for k, v in CLASS_TO_MED.items()}

def label_to_medication(label: str) -> str:
    key = normalize_name(label)
    if key in CLASS_TO_MED_NORM:
        return CLASS_TO_MED_NORM[key]
    return label

class PillVision:
    """
    Handles:
    - YOLO pill detection
    - Face recognition
    - Scanning all tray positions
    - Picking pills for a given patient & time slot
    """
    def __init__(self, frame_provider, patient_db, arm_controller,
                 yolo_weights="/home/pi/Downloads/best (4).pt"):
        """
        frame_provider: function returning the latest camera frame (or None)
        patient_db:     PatientDatabase instance
        arm_controller: ArmController instance
        """
        self.get_frame = frame_provider
        self.patient_db = patient_db
        self.arm = arm_controller
        self.model = YOLO(yolo_weights)

    # camera helper

    def capture_clear_image(self, position_name):
        print(f"[CAMERA] Capturing frame at {position_name}...")
        time.sleep(1.0)

        frame = self.get_frame()
        if frame is not None:
            frame = cv2.resize(frame, (640, 480))
            print("[CAMERA] Frame captured")
            return frame
        else:
            print("[CAMERA] No frame available (camera not ready?)")
            return None

    # YOLO detection

    def yolo_detection(self, img):
        results = self.model(img, verbose=False)
        detections = []

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                label = self.model.names[cls]
                detections.append((label, conf, (x1, y1, x2, y2)))

                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, f"{label} {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        return img, detections

    # FACE DETECTION

    def detect_patient(self, img):
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        face_locs = face_recognition.face_locations(rgb)
        face_encs = face_recognition.face_encodings(rgb, face_locs)

        recognized_patients = []

        for idx, (top, right, bottom, left) in enumerate(face_locs):
            if idx >= len(face_encs):
                continue
            face_encoding = face_encs[idx]

            matches = face_recognition.compare_faces(
                self.patient_db.face_encodings,
                face_encoding,
                tolerance=0.5
            )

            name = "Unknown"
            patient_id = None
            patient_data = None

            if True in matches:
                first_match_index = matches.index(True)
                patient_id = self.patient_db.patient_names[first_match_index]
                patient_data = self.patient_db.patients.get(patient_id)
                if patient_data:
                    name = patient_data["full_name"]

            color = (0, 255, 0) if patient_id else (0, 0, 255)
            cv2.rectangle(img, (left, top), (right, bottom), color, 2)

            info_line1 = f"Patient: {name}"
            cv2.putText(img, info_line1, (left, top - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            if patient_data:
                info_line2 = f"Condition: {patient_data['condition']}"
                cv2.putText(img, info_line2, (left, top - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                recognized_patients.append({
                    "id": patient_id,
                    "name": name,
                    "condition": patient_data["condition"],
                    "medications": patient_data["medications"],
                    "schedule": patient_data.get("schedule", {})
                })

        return img, recognized_patients

    # SCAN PILL POSITIONS

    def scan_pill_positions(self, patient_name):
        pill_positions = [
            ("Right_1", self.arm.P_PILL_1, self.arm.P_PICK_1),
            ("Right_2", self.arm.P_PILL_2, self.arm.P_PICK_2),
            ("Right_3", self.arm.P_PILL_3, self.arm.P_PICK_3),
            ("Left_1",  self.arm.P_PILL_4, self.arm.P_PICK_4),
            ("Left_2",  self.arm.P_PILL_5, self.arm.P_PICK_5),
            ("Left_3",  self.arm.P_PILL_6, self.arm.P_PICK_6),
        ]

        found_pills = []

        print("\n" + "=" * 60)
        print(f"SCANNING PILL POSITIONS FOR {patient_name.upper()}")
        print("=" * 60)

        self.arm.open_gripper()
        self.arm.move_joints(self.arm.P_ABOVE_PILL[:5], 1000)

        for position_name, pill_scan_pos, pill_pick_pos in pill_positions:
            print(f"\n CHECKING {position_name}")
            print("-" * 30)

            self.arm.move_joints(pill_scan_pos[:5], 1000)
            time.sleep(0.5)

            frame = self.capture_clear_image(position_name)
            if frame is not None:
                raw_path = f"/home/pi/Desktop/scan_{position_name}.jpg"
                cv2.imwrite(raw_path, frame)
                print(f"[SAVE] Raw scan saved: {raw_path}")

                detected_img, detections = self.yolo_detection(frame.copy())
                det_path = f"/home/pi/Desktop/scan_{position_name}_detected.jpg"
                cv2.imwrite(det_path, detected_img)
                print(f"[SAVE] Detection image saved: {det_path}")

                if detections:
                    print(f" FOUND {len(detections)} pill(s)")
                    for label, conf, bbox in detections:
                        print(f"   • {label} ({conf:.2f})")

                    found_pills.append({
                        "position": position_name,
                        "scan_pos": pill_scan_pos,
                        "pick_pos": pill_pick_pos,
                        "detections": detections
                    })
                else:
                    print(" No pills detected at this spot.")

            self.arm.move_joints(self.arm.P_ABOVE_PILL[:5], 1000)
            time.sleep(0.3)

        return found_pills

    # PICK PILLS PER TIME SLOT

    def pick_all_detected_pills(self, found_pills, patient, slot_key):
        patient_name = patient["name"]
        schedule = patient.get("schedule", {})
        slot_meds = schedule.get(slot_key, patient["medications"])

        if not slot_meds:
            print(f"\n[PICK] For {patient_name} at {slot_key}, no meds are scheduled.")
            return 0, []

        needed_meds_norm = {normalize_name(m): m for m in slot_meds}
        needed_meds_display = list(needed_meds_norm.values())

        if not found_pills:
            print("\n[PICK] No pills to pick up.")
            return 0, []

        print("\n" + "=" * 60)
        print(f"FILTERING PILLS FOR {patient_name.upper()} at time slot {slot_key}")
        print(f"Medications required at this slot: {', '.join(needed_meds_display)}")
        print("=" * 60)

        filtered_pills = []
        for pill_info in found_pills:
            detections = pill_info["detections"]
            meds_here = set()

            for (label, conf, bbox) in detections:
                med_name_raw = label_to_medication(label)
                med_key = normalize_name(med_name_raw)
                if med_key in needed_meds_norm:
                    meds_here.add(needed_meds_norm[med_key])

            if meds_here:
                pill_info["meds_for_patient"] = list(meds_here)
                filtered_pills.append(pill_info)

        if not filtered_pills:
            print("\n[PICK] None of the trays match this patient's meds for this time slot.")
            return 0, []

        print(f"[PICK] {len(filtered_pills)} tray position(s) have the right meds.")
        for fp in filtered_pills:
            print(f"  - {fp['position']} -> {fp.get('meds_for_patient', [])}")

        successful_picks = 0
        used_positions = set()
        tray_moves = []

        for med_index, med in enumerate(slot_meds):
            print(f"\n[PICK] Searching for a tray containing: {med}")

            candidate = None
            for pill_info in filtered_pills:
                pos_name = pill_info["position"]
                if pos_name in used_positions:
                    continue
                meds_for_pos = pill_info.get("meds_for_patient", [])
                if med in meds_for_pos:
                    candidate = pill_info
                    break

            if candidate is None:
                print(f"[PICK] ⚠️ No tray found for medication: {med}")
                continue

            pos_name = candidate["position"]
            pick_pos = candidate["pick_pos"]
            meds_for_pos = candidate.get("meds_for_patient", [])

            if med_index == 0:
                drop_position = self.arm.P_CENTER_PLACE
            elif med_index == 1:
                drop_position = self.arm.P_CENTER_PLACE_2
            else:
                drop_position = self.arm.P_CENTER_PLACE_2

            print(f"[PICK] -> Using tray {pos_name} with meds: {', '.join(meds_for_pos)}")
            print(f"[PICK] -> Dropping at: "
                  f"{'P_CENTER_PLACE' if med_index == 0 else 'P_CENTER_PLACE_2'}")

            success = self.arm.pick_tray(pos_name, pick_pos, drop_position)
            if success:
                successful_picks += 1
                used_positions.add(pos_name)
                tray_moves.append({
                    "position": pos_name,
                    "pick_pos": pick_pos,
                    "drop_pos": drop_position
                })
                print(f"[PICK] Completed pick for medication: {med}")
            else:
                print(f"[PICK] Could not pick medication: {med}")

            time.sleep(1)

        print(f"\n[PICK] SUMMARY for {patient_name} at {slot_key}: "
              f"{successful_picks} tray(s) picked for {len(slot_meds)} medication(s).")
        return successful_picks, tray_moves