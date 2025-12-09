#!/usr/bin/env python3
# coding=utf-8

import cv2
import time
import json
import os
import threading
import builtins
from datetime import datetime

import customtkinter as ctk
from PIL import Image, ImageTk

from Arm_Lib import Arm_Device
import face_recognition
from ultralytics import YOLO

# ------------------------------------------------
# GLOBALS FOR GUI / CAMERA
# ------------------------------------------------
latest_frame = None
camera_running = False
cap = None
log_callback = None  # set by GUI to receive log lines

# Intercept all print() calls and forward to GUI log
_real_print = builtins.print
def gui_print(*args, **kwargs):
    s = " ".join(str(a) for a in args)
    _real_print(*args, **kwargs)
    if log_callback:
        log_callback(s)
builtins.print = gui_print


# ------------------------------------------------
# DOFBOT INIT
# ------------------------------------------------
Arm = Arm_Device()
time.sleep(0.1)

# ------------------------------------------------
# ARM POSITIONS
# ------------------------------------------------
JOINTS_HOME = [90, 135, 20, 25, 90, 30]           # Home position
JOINTS_SCAN_HEIGHT = [90, 100, 0, 40, 90, 30]     # (Not directly used)

# Pill inspection positions (close to table to see pills clearly)
# Right side
P_PILL_1 = [10, 138, -35, 14, 269, 30]   # Right 1
P_PILL_2 = [40, 138, -35, 14, 269, 30]   # Right 2
P_PILL_3 = [70, 142, -35, 18, 269, 30]   # Right 3

# Left side
P_PILL_4 = [174, 138, -35, 14, 269, 30]  # Left 1
P_PILL_5 = [140, 138, -35, 14, 269, 30]  # Left 2
P_PILL_6 = [110, 142, -35, 18, 269, 30]  # Left 3

# Pill picking positions (slightly lower for grabbing)
# Right
P_PICK_1 = [10, 66, 15, 33, 270, 30]     # Right 1
P_PICK_2 = [44, 66, 20, 28, 270, 30]     # Right 2
P_PICK_3 = [70, 66, 15, 33, 270, 30]     # Right 3

# Left
P_PICK_4 = [170, 66, 15, 33, 270, 30]    # Left 1
P_PICK_5 = [140, 66, 20, 28, 270, 30]    # Left 2
P_PICK_6 = [110, 66, 15, 33, 270, 30]    # Left 3

# Center placement positions (drop trays/pills)
P_CENTER_PLACE   = [90, 66, 20, 29, 270, 30]  # First slot
P_CENTER_PLACE_2 = [90, 88, 20, 29, 270, 30]  # Second slot (next/below)

# Safe position above pills (for moving between positions)
P_ABOVE_PILL = [90, 80, 50, 50, 270, 30]

# Move to home at start
Arm.Arm_serial_servo_write6_array(JOINTS_HOME, 1500)
time.sleep(2)

# ------------------------------------------------
# GRIPPER CONFIG
# ------------------------------------------------
GRIPPER_OPEN_ANGLE = 60
GRIPPER_CLOSE_ANGLE = 135  # increase if you want stronger grip

def arm_clamp_block(enable):
    if enable == 0:
        Arm.Arm_serial_servo_write(6, GRIPPER_OPEN_ANGLE, 400)  # Open
    else:
        Arm.Arm_serial_servo_write(6, GRIPPER_CLOSE_ANGLE, 400) # Close
    time.sleep(.5)

def gripper_open():
    arm_clamp_block(0)

def gripper_close():
    arm_clamp_block(1)

# ------------------------------------------------
# BASIC ARM MOVES
# ------------------------------------------------
def arm_move(p, s_time=500):
    """Move arm joints 1-5"""
    for i in range(5):
        sid = i + 1
        if sid == 5:
            time.sleep(.1)
            Arm.Arm_serial_servo_write(sid, p[i], int(s_time * 1.2))
        else:
            Arm.Arm_serial_servo_write(sid, p[i], s_time)
        time.sleep(.01)
    time.sleep(s_time / 1000)

def arm_move_up():
    Arm.Arm_serial_servo_write(2, 90, 1500)
    Arm.Arm_serial_servo_write(3, 90, 1500)
    Arm.Arm_serial_servo_write(4, 90, 1500)
    time.sleep(0.1)

# ------------------------------------------------
# LIFT & DROP TUNING
# ------------------------------------------------
TRANSPORT_LIFT_J3 = 35    # how high to lift before moving sideways
DROP_DOWN_J3      = -14   # how much to lower at drop (adjust sign if needed)

# ------------------------------------------------
# CAMERA FRAME ACCESS
# ------------------------------------------------
def get_latest_frame():
    global latest_frame
    if latest_frame is None:
        return None
    return latest_frame.copy()

# ------------------------------------------------
# SIMPLE PICK ROUTINE
# ------------------------------------------------
def pick_pill_simple(position_name, pick_position, drop_position):
    """
    pick_position: where the tray/pill is picked from (P_PICK_x)
    drop_position: where the tray is dropped (P_CENTER_PLACE / P_CENTER_PLACE_2)
    """
    print(f"\n[PICK] 🟢 Starting pick from {position_name}")
    print("-" * 40)

    EXTRA_DOWN_J2 = 0
    EXTRA_DOWN_J3 = 0
    EXTRA_DOWN_J4 = 0

    try:
        print("[PICK] Step 1: Open gripper")
        gripper_open()
        time.sleep(0.5)

        print("[PICK] Step 2: Move to safe height")
        safe_pos = P_ABOVE_PILL[:5]
        arm_move(safe_pos, 800)
        time.sleep(0.5)

        print("[PICK] Step 3: Move down to pill/tray")
        pick_pos_joints = pick_position[:5]
        arm_move(pick_pos_joints, 1000)
        time.sleep(0.5)

        if any([EXTRA_DOWN_J2, EXTRA_DOWN_J3, EXTRA_DOWN_J4]):
            print("[PICK] Step 3b: Touch down closer to table")
            touch_down = pick_pos_joints.copy()
            touch_down[1] += EXTRA_DOWN_J2
            touch_down[2] += EXTRA_DOWN_J3
            touch_down[3] += EXTRA_DOWN_J4
            arm_move(touch_down, 700)
            time.sleep(0.5)
            current_pos = touch_down
        else:
            current_pos = pick_pos_joints

        print("[PICK] Step 4: Close gripper")
        gripper_close()
        time.sleep(0.8)

        print("[PICK] Step 5: Lift pill/tray higher for transport")
        lift_pos = current_pos.copy()
        lift_pos[2] += TRANSPORT_LIFT_J3
        arm_move(lift_pos, 800)
        time.sleep(0.5)

        print("[PICK] Step 6: Move to drop position")
        drop_joints = drop_position[:5]
        arm_move(drop_joints, 800)
        time.sleep(0.5)

        print("[PICK] Step 6b: Lower to tray height")
        final_drop = drop_joints.copy()
        final_drop[2] += DROP_DOWN_J3
        arm_move(final_drop, 500)
        time.sleep(0.3)

        print("[PICK] Step 7: Open gripper to drop")
        gripper_open()
        time.sleep(0.5)

        print("[PICK] Step 8: Clear from drop position")
        clear_pos = final_drop.copy()
        clear_pos[2] += 15
        arm_move(clear_pos, 600)
        time.sleep(0.5)

        print("[PICK] Step 9: Return to safe height")
        arm_move(safe_pos, 800)

        print(f"[PICK] ✅ Successfully picked from {position_name}")
        return True

    except Exception as e:
        print(f"[PICK] ❌ Failed to pick from {position_name}: {e}")
        try:
            arm_move_up()
            gripper_open()
        except:
            pass
        return False

# ------------------------------------------------
# RETURN TRAYS TO ORIGINAL P_PICK POSITIONS
# ------------------------------------------------
def return_trays(tray_moves):
    if not tray_moves:
        print("\n[TRAYS] No trays to return.")
        return

    print("\n" + "=" * 60)
    print("RETURNING TRAYS TO ORIGINAL POSITIONS")
    print("=" * 60)

    for i, move_info in enumerate(tray_moves, start=1):
        pos_name = move_info["position"]
        pick_pos = move_info["pick_pos"]
        drop_pos = move_info["drop_pos"]

        print(f"\n[TRAY] Returning tray {i}/{len(tray_moves)} from center back to {pos_name}")

        safe_pos = P_ABOVE_PILL[:5]
        arm_move(safe_pos, 800)
        time.sleep(0.5)

        center_transport = drop_pos[:5].copy()
        center_transport[2] += TRANSPORT_LIFT_J3
        print("[TRAY] Moving above tray at center")
        arm_move(center_transport, 800)
        time.sleep(0.5)

        print("[TRAY] Lowering to grip tray")
        center_grip = drop_pos[:5].copy()
        center_grip[2] += DROP_DOWN_J3
        arm_move(center_grip, 700)
        time.sleep(0.3)

        print("[TRAY] Closing gripper to hold tray")
        gripper_close()
        time.sleep(0.5)

        print("[TRAY] Lifting tray for transport")
        lift_from_center = center_grip.copy()
        lift_from_center[2] += TRANSPORT_LIFT_J3
        arm_move(lift_from_center, 800)
        time.sleep(0.5)

        print(f"[TRAY] Moving over original position: {pos_name}")
        pick_transport = pick_pos[:5].copy()
        pick_transport[2] += TRANSPORT_LIFT_J3
        arm_move(pick_transport, 800)
        time.sleep(0.5)

        print("[TRAY] Lowering tray back to its original place")
        arm_move(pick_pos[:5], 700)
        time.sleep(0.3)

        print("[TRAY] Releasing tray")
        gripper_open()
        time.sleep(0.5)

        clear_pick = pick_pos[:5].copy()
        clear_pick[2] += 15
        arm_move(clear_pick, 700)
        time.sleep(0.5)

    arm_move(P_ABOVE_PILL[:5], 800)
    print("\n[TRAYS] ✅ All trays returned to their original positions.")

# ------------------------------------------------
# PATIENT DATABASE WITH TIME SLOTS
# ------------------------------------------------
class PatientDatabase:
    def __init__(self):
        self.patients = {}
        self.face_encodings = []
        self.patient_names = []
        self.load_patients()

    def load_patients(self):
        patients_file = "/home/pi/Desktop/patients.json"

        if os.path.exists(patients_file):
            try:
                with open(patients_file, 'r') as f:
                    self.patients = json.load(f)
                print(f"[INFO] Loaded {len(self.patients)} patients")
            except:
                print("[WARNING] Could not load patients file, using defaults.")
                self.create_default_patients()
        else:
            self.create_default_patients()
            self.save_patients()

        self.load_face_encodings()

    def create_default_patients(self):
        self.patients = {
            "nahida": {
                "full_name": "Nahida",
                "condition": "Diabetes",
                "medications": ["Vildaril", "Clopica"],
                "schedule": {
                    "09:00": ["Vildaril"],
                    "12:00": ["Clopica"],
                    "14:00": ["Vildaril", "Clopica"]
                },
                "face_image": "/home/pi/Downloads/nahida.jpeg",
                "face_encoding": None,
                "last_dispensed": {}
            },
            "adrian": {
                "full_name": "Adrian",
                "condition": "Joint Pain",
                "medications": ["Osteocare", "Flucoxacilin"],
                "schedule": {
                    "09:00": ["Osteocare"],
                    "12:00": ["Flucoxacilin"],
                    "14:00": []
                },
                "face_image": "/home/pi/Downloads/adrian.jpeg",
                "face_encoding": None,
                "last_dispensed": {}
            },
            "roshni": {
                "full_name": "Roshni",
                "condition": "Cholesterol",
                "medications": ["Imdex", "Doliprane"],
                "schedule": {
                    "09:00": ["Imdex"],
                    "12:00": ["Doliprane"],
                    "14:00": []
                },
                "face_image": "/home/pi/Downloads/Roshni.jpeg",
                "face_encoding": None,
                "last_dispensed": {}
            }
        }

    def load_face_encodings(self):
        self.face_encodings = []
        self.patient_names = []

        for patient_id, data in self.patients.items():
            face_image_path = data.get("face_image", "")
            if os.path.exists(face_image_path):
                try:
                    img = face_recognition.load_image_file(face_image_path)
                    encoding_list = face_recognition.face_encodings(img)
                    if encoding_list:
                        encoding = encoding_list[0]
                        self.face_encodings.append(encoding)
                        self.patient_names.append(patient_id)
                        data["face_encoding"] = None
                        print(f"[INFO] Loaded face for {data['full_name']}")
                except:
                    print(f"[WARNING] Could not load face for {patient_id}")

    def save_patients(self):
        with open("/home/pi/Desktop/patients.json", 'w') as f:
            json.dump(self.patients, f, indent=2)

    def record_dispense(self, patient_id, slot_key):
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if patient_id in self.patients:
            self.patients[patient_id].setdefault("last_dispensed", {})
            self.patients[patient_id]["last_dispensed"][slot_key] = now_str
            self.save_patients()
            print(f"[INFO] Recorded dispense for {patient_id} at slot {slot_key} -> {now_str}")

patient_db = PatientDatabase()

# ------------------------------------------------
# YOLO INIT
# ------------------------------------------------
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
            cv2.putText(img, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return img, detections

# ------------------------------------------------
# NAME NORMALIZATION & MAPPING
# ------------------------------------------------
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

# ------------------------------------------------
# FACE DETECTION
# ------------------------------------------------
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

# ------------------------------------------------
# CAPTURE FUNCTIONS
# ------------------------------------------------
def capture_clear_image(position_name):
    print(f"[CAMERA] Preparing to capture at {position_name}...")
    time.sleep(1.0)

    frame = get_latest_frame()
    if frame is not None:
        frame = cv2.resize(frame, (640, 480))
        print("[CAMERA] ✅ Frame captured")
        return frame
    else:
        print("[CAMERA] ❌ No frame available")
        return None

# ------------------------------------------------
# SCAN PILL POSITIONS
# ------------------------------------------------
def scan_pill_positions(patient_name):
    pill_positions = [
        ("Right_1", P_PILL_1, P_PICK_1),
        ("Right_2", P_PILL_2, P_PICK_2),
        ("Right_3", P_PILL_3, P_PICK_3),
        ("Left_1",  P_PILL_4, P_PICK_4),
        ("Left_2",  P_PILL_5, P_PICK_5),
        ("Left_3",  P_PILL_6, P_PICK_6),
    ]

    found_pills = []

    print("\n" + "=" * 60)
    print(f"SCANNING PILL POSITIONS FOR {patient_name.upper()}")
    print("=" * 60)

    gripper_open()
    arm_move(P_ABOVE_PILL[:5], 1000)

    for position_name, pill_scan_pos, pill_pick_pos in pill_positions:
        print(f"\n📍 CHECKING {position_name}")
        print("-" * 30)

        arm_move(pill_scan_pos[:5], 1000)
        time.sleep(0.5)

        frame = capture_clear_image(position_name)
        if frame is not None:
            raw_path = f"/home/pi/Desktop/scan_{position_name}.jpg"
            cv2.imwrite(raw_path, frame)
            print(f"[SAVE] Raw scan saved: {raw_path}")

            detected_img, detections = yolo_detection(frame.copy())
            det_path = f"/home/pi/Desktop/scan_{position_name}_detected.jpg"
            cv2.imwrite(det_path, detected_img)
            print(f"[SAVE] Detection image saved: {det_path}")

            if detections:
                print(f"✅ FOUND {len(detections)} pill(s)")
                for label, conf, bbox in detections:
                    print(f"   • {label} ({conf:.2f})")

                found_pills.append({
                    "position": position_name,
                    "scan_pos": pill_scan_pos,
                    "pick_pos": pill_pick_pos,
                    "detections": detections
                })
            else:
                print("❌ NO pills found")

        arm_move(P_ABOVE_PILL[:5], 1000)
        time.sleep(0.3)

    return found_pills

# ------------------------------------------------
# PICK PILLS PER TIME SLOT
# ------------------------------------------------
def pick_all_detected_pills(found_pills, patient, slot_key):
    patient_name = patient["name"]
    schedule = patient.get("schedule", {})
    slot_meds = schedule.get(slot_key, patient["medications"])

    if not slot_meds:
        print(f"\n[PICK] For {patient_name} at {slot_key}, no meds scheduled.")
        return 0, []

    needed_meds_norm = {normalize_name(m): m for m in slot_meds}
    needed_meds_display = list(needed_meds_norm.values())

    if not found_pills:
        print("\n[PICK] No pills to pick up")
        return 0, []

    print("\n" + "=" * 60)
    print(f"FILTERING PILLS FOR {patient_name.upper()} at time slot {slot_key}")
    print(f"Patient needs at this slot: {', '.join(needed_meds_display)}")
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
        print("\n[PICK] ❌ None of the detected pills match this patient's meds for this slot.")
        return 0, []

    print(f"[PICK] ✅ {len(filtered_pills)} pill position(s) contain meds for {patient_name}.")
    for fp in filtered_pills:
        print(f"  - {fp['position']} -> {fp.get('meds_for_patient', [])}")

    successful_picks = 0
    used_positions = set()
    tray_moves = []

    for med_index, med in enumerate(slot_meds):
        print(f"\n[PICK] Looking for a pill with medication: {med}")

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
            print(f"[PICK] ⚠️ No available pill position found for medication: {med}")
            continue

        pos_name = candidate["position"]
        pick_pos = candidate["pick_pos"]
        meds_for_pos = candidate.get("meds_for_patient", [])

        if med_index == 0:
            drop_position = P_CENTER_PLACE
        elif med_index == 1:
            drop_position = P_CENTER_PLACE_2
        else:
            drop_position = P_CENTER_PLACE_2

        print(f"[PICK] -> Using position {pos_name} which has meds: {', '.join(meds_for_pos)}")
        print(f"[PICK] -> Will drop to {'P_CENTER_PLACE' if med_index == 0 else 'P_CENTER_PLACE_2'}")

        success = pick_pill_simple(pos_name, pick_pos, drop_position)
        if success:
            successful_picks += 1
            used_positions.add(pos_name)
            tray_moves.append({
                "position": pos_name,
                "pick_pos": pick_pos,
                "drop_pos": drop_position
            })
            print(f"[PICK] ✅ Successfully picked for medication: {med}")
        else:
            print(f"[PICK] ❌ Failed to pick for medication: {med}")

        time.sleep(1)

    print(f"\n[PICK] SUMMARY for {patient_name} at {slot_key}: "
          f"{successful_picks} pill(s) picked for {len(slot_meds)} scheduled meds.")
    return successful_picks, tray_moves

# ------------------------------------------------
# ROBOT MAIN LOGIC (GUI MODE)
# ------------------------------------------------
def robot_main(time_slot_key):
    print("[INFO] Medical Assistant DOFBOT Initialized (GUI mode)")
    print(f"[INFO] Using time slot: {time_slot_key}")
    print("[INFO] Looking for patients...")

    gripper_open()

    current_patient_id = None
    patient_detection_time = None
    detection_confidence_time = 3.0

    try:
        while True:
            frame = get_latest_frame()
            if frame is None:
                time.sleep(0.1)
                continue

            frame = cv2.resize(frame, (640, 480))
            display_frame = frame.copy()

            display_frame, patients = detect_patient(display_frame)

            if patients:
                patient = patients[0]
                pid = patient["id"]

                if current_patient_id != pid:
                    current_patient_id = pid
                    patient_detection_time = time.time()
                    print(f"\n[FACE] Detected: {patient['name']}")
                    print(f"[FACE] All medications: {', '.join(patient['medications'])}")
                    print(f"[FACE] Time-slot meds at {time_slot_key}: "
                          f"{', '.join(patient.get('schedule', {}).get(time_slot_key, patient['medications']))}")

                elif (current_patient_id == pid and
                      patient_detection_time and
                      time.time() - patient_detection_time >= detection_confidence_time):

                    print("\n" + "=" * 60)
                    print(f"✅ PATIENT CONFIRMED: {patient['name']} for slot {time_slot_key}")
                    print("=" * 60)

                    found_pills = scan_pill_positions(patient['name'])

                    if found_pills:
                        print(f"\n✅ Found pills at {len(found_pills)} position(s)")
                        picked_count, tray_moves = pick_all_detected_pills(found_pills, patient, time_slot_key)
                        if picked_count > 0:
                            patient_db.record_dispense(patient['id'], time_slot_key)
                    else:
                        print("\n❌ No pills found")
                        picked_count, tray_moves = 0, []

                    print("\n[ACTION] Moving to safe height above trays...")
                    arm_move(P_ABOVE_PILL[:5], 1000)
                    gripper_open()

                    print("\n[TRAYS] Please remove pills from trays at center positions,")
                    print("        then place EMPTY trays back at the same center positions.")
                    input("[TRAYS] When trays are ready to return, press ENTER in console...")

                    return_trays(tray_moves)

                    print("\n[ACTION] Returning to home position...")
                    arm_move(JOINTS_HOME[:5], 1000)
                    gripper_open()

                    answer = input("\nDo you want to scan for the next patient? (y/n): ").strip().lower()
                    if answer not in ("y", "yes"):
                        print("[INFO] Stopping patient scanning.")
                        break
                    else:
                        print("\n[READY] Ready for next patient...")
                        current_patient_id = None
                        patient_detection_time = None
                        time.sleep(2)
                        continue

            else:
                current_patient_id = None
                patient_detection_time = None

            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\n[INFO] Program interrupted by user")
    finally:
        arm_move(JOINTS_HOME[:5], 1000)
        gripper_open()
        print("[INFO] Robot logic finished for this run.")

# ------------------------------------------------
# GUI WITH customtkinter
# ------------------------------------------------
class DofbotGUI(ctk.CTk):
    def __init__(self):
        super().__init__()

        global log_callback
        log_callback = self.append_log

        self.title("DOFBOT Pill Assistant - GUI")
        self.geometry("1100x650")

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.time_slot_var = ctk.StringVar(value="09:00")

        # Top controls frame
        controls = ctk.CTkFrame(self)
        controls.pack(side="top", fill="x", padx=10, pady=10)

        title_label = ctk.CTkLabel(
            controls,
            text="DOFBOT Smart Pill Assistant",
            font=ctk.CTkFont(size=20, weight="bold")
        )
        title_label.pack(side="left", padx=10)

        slot_label = ctk.CTkLabel(controls, text="Time slot:")
        slot_label.pack(side="left", padx=(40, 5))

        self.slot_menu = ctk.CTkOptionMenu(
            controls,
            variable=self.time_slot_var,
            values=["09:00", "12:00", "14:00"]
        )
        self.slot_menu.pack(side="left", padx=5)

        self.start_button = ctk.CTkButton(
            controls,
            text="Start Robot",
            command=self.start_robot_clicked
        )
        self.start_button.pack(side="left", padx=(20, 5))

        # NEW: Exit button to close GUI + release camera
        self.exit_button = ctk.CTkButton(
            controls,
            text="Exit",
            fg_color="#AA3333",
            hover_color="#882222",
            command=self.close_app
        )
        self.exit_button.pack(side="right", padx=(5, 10))

        # Main content frame
        main_frame = ctk.CTkFrame(self)
        main_frame.pack(fill="both", expand=True, padx=10, pady=5)

        # Left: camera
        self.camera_label = ctk.CTkLabel(main_frame, text="Camera feed will appear here")
        self.camera_label.pack(side="left", fill="both", expand=True, padx=(10, 5), pady=10)

        # Right: log text
        log_frame = ctk.CTkFrame(main_frame)
        log_frame.pack(side="right", fill="both", expand=True, padx=(5, 10), pady=10)

        log_label = ctk.CTkLabel(
            log_frame,
            text="Console / Status",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        log_label.pack(anchor="w", padx=5, pady=(5, 0))

        self.log_text = ctk.CTkTextbox(log_frame, wrap="word")
        self.log_text.pack(fill="both", expand=True, padx=5, pady=5)
        self.log_text.insert("end", "[INFO] GUI ready.\n")
        self.log_text.configure(state="disabled")

        self.camera_thread_started = False
        self.robot_thread_running = False

        self.after(30, self.update_camera_frame)

        # When user clicks the window X, also clean up camera
        self.protocol("WM_DELETE_WINDOW", self.close_app)

    # ---------- logging ----------
    def append_log(self, line: str):
        self.log_text.configure(state="normal")
        self.log_text.insert("end", line + "\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    # ---------- camera handling ----------
    def start_camera(self):
        global cap, camera_running
        if camera_running:
            return
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("[ERROR] Cannot open camera from GUI.")
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        camera_running = True
        print("[INFO] Camera started from GUI.")

    def stop_camera(self):
        global cap, camera_running
        if cap is not None:
            cap.release()
            cap = None
        camera_running = False
        print("[INFO] Camera stopped and released.")

    def update_camera_frame(self):
        global latest_frame, camera_running, cap

        if camera_running and cap is not None:
            ret, frame = cap.read()
            if ret:
                latest_frame = frame
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(rgb)
                img = img.resize((520, 390))
                imgtk = ImageTk.PhotoImage(image=img)
                self.camera_label.configure(image=imgtk, text="")
                self.camera_label.image = imgtk
        self.after(30, self.update_camera_frame)

    # ---------- robot start ----------
    def start_robot_clicked(self):
        if self.robot_thread_running:
            print("[INFO] Robot already running.")
            return

        self.start_camera()

        slot = self.time_slot_var.get()
        print(f"[GUI] Starting robot for time slot {slot}")
        self.robot_thread_running = True
        self.start_button.configure(state="disabled", text="Running...")

        t = threading.Thread(target=self._run_robot_thread, args=(slot,), daemon=True)
        t.start()

    def _run_robot_thread(self, slot_key):
        try:
            robot_main(slot_key)
        finally:
            self.robot_thread_running = False
            print("[GUI] Robot thread finished.")
            # Make sure we stop camera & close GUI cleanly after run
            self.after(0, self.close_app)

    # ---------- close app (stop camera + destroy window) ----------
    def close_app(self):
        """Stop camera, release resources, and close GUI window."""
        self.stop_camera()
        try:
            self.destroy()
        except:
            pass
        os._exit(0)


# ------------------------------------------------
# ENTRY POINT
# ------------------------------------------------
if __name__ == "__main__":
    app = DofbotGUI()
    app.mainloop()