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

import speech_recognition as sr   # Speech recognition

#SPEECH RECOGNITION CONFIG 
recognizer = sr.Recognizer()

MIC_INDEX = 3  

#GLOBALS FOR GUI / CAMERA / ROBOT <-> UI LINK
latest_frame = None
camera_running = False
cap = None
log_callback = None

# These sync points replace input() in the terminal:
trays_ready_event = threading.Event()
next_patient_event = threading.Event()
next_patient_decision = "stop"

# GUI instance so robot_main can nudge the interface
gui_instance = None

# Intercept print() so logs appear both in terminal and in GUI
_real_print = builtins.print
def gui_print(*args, **kwargs):
    s = " ".join(str(a) for a in args)
    _real_print(*args, **kwargs)
    if log_callback:
        log_callback(s)
builtins.print = gui_print


# DOFBOT INIT
Arm = Arm_Device()
time.sleep(0.1)

# ARM POSITIONS
JOINTS_HOME = [90, 135, 20, 25, 90, 30]           # Home / neutral pose
JOINTS_SCAN_HEIGHT = [90, 100, 0, 40, 90, 30]     # Not used directly for now

# Pill inspection positions (camera looking at trays)
# Right side
P_PILL_1 = [10, 138, -35, 14, 269, 30]   # Right 1
P_PILL_2 = [40, 138, -35, 14, 269, 30]   # Right 2
P_PILL_3 = [70, 142, -35, 18, 269, 30]   # Right 3

# Left side
P_PILL_4 = [174, 138, -35, 14, 269, 30]  # Left 1
P_PILL_5 = [140, 138, -35, 14, 269, 30]  # Left 2
P_PILL_6 = [110, 142, -35, 18, 269, 30]  # Left 3

# Pill picking positions (slightly lower than the inspection positions)
# Right
P_PICK_1 = [10, 66, 15, 33, 270, 30]     # Right 1
P_PICK_2 = [44, 66, 20, 28, 270, 30]     # Right 2
P_PICK_3 = [70, 66, 15, 33, 270, 30]     # Right 3

# Left
P_PICK_4 = [170, 66, 15, 33, 270, 30]    # Left 1
P_PICK_5 = [140, 66, 20, 28, 270, 30]    # Left 2
P_PICK_6 = [110, 66, 15, 33, 270, 30]    # Left 3

# Center placement positions (where the trays are dropped)
P_CENTER_PLACE   = [90, 66, 20, 29, 270, 30]  # First slot
P_CENTER_PLACE_2 = [90, 88, 20, 29, 270, 30]  # Second slot

# Safe position above table to move around without hitting anything
P_ABOVE_PILL = [90, 80, 50, 50, 270, 30]

# Move to home on startup
Arm.Arm_serial_servo_write6_array(JOINTS_HOME, 1500)
time.sleep(2)

# GRIPPER CONFIG
GRIPPER_OPEN_ANGLE = 60
GRIPPER_CLOSE_ANGLE = 145  # increase if you want the clamp to squeeze harder

def arm_clamp_block(enable):
    if enable == 0:
        Arm.Arm_serial_servo_write(6, GRIPPER_OPEN_ANGLE, 400)  # open
    else:
        Arm.Arm_serial_servo_write(6, GRIPPER_CLOSE_ANGLE, 400) # close
    time.sleep(.5)

def gripper_open():
    arm_clamp_block(0)

def gripper_close():
    arm_clamp_block(1)

# BASIC ARM MOVE HELPERS
def arm_move(p, s_time=500):
    """Send angles to servos 1–5 with a small stagger for servo 5."""
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
    """Bring the arm up to a more neutral pose (used for recovery)."""
    Arm.Arm_serial_servo_write(2, 90, 1500)
    Arm.Arm_serial_servo_write(3, 90, 1500)
    Arm.Arm_serial_servo_write(4, 90, 1500)
    time.sleep(0.1)

# LIFT & DROP TUNING
TRANSPORT_LIFT_J3 = 50   # how much we lift at joint 3 when travelling
DROP_DOWN_J3      = 0    # how much we lower at drop so the tray doesn’t fall from too high
RETURN_PICK_DOWN_J3 = 50 # how much we lift when returning trays to original spots

# CAMERA FRAME ACCESS
def get_latest_frame():
    global latest_frame
    if latest_frame is None:
        return None
    return latest_frame.copy()

# SIMPLE PICK ROUTINE
def pick_pill_simple(position_name, pick_position, drop_position):
    """
    Take one tray/pill from its pick position and drop it at a chosen center slot.
    """
    print(f"\n[PICK] Starting pick from {position_name}")
    print("-" * 40)

    EXTRA_DOWN_J2 = 0
    EXTRA_DOWN_J3 = 0
    EXTRA_DOWN_J4 = 0

    try:
        print("[PICK] Step 1: Open gripper")
        gripper_open()
        time.sleep(0.5)

        print("[PICK] Step 2: Move to safe height above the table")
        safe_pos = P_ABOVE_PILL[:5]
        arm_move(safe_pos, 800)
        time.sleep(0.5)

        print("[PICK] Step 3: Move down to tray")
        pick_pos_joints = pick_position[:5]
        arm_move(pick_pos_joints, 1000)
        time.sleep(0.5)

        if any([EXTRA_DOWN_J2, EXTRA_DOWN_J3, EXTRA_DOWN_J4]):
            print("[PICK] Step 3b: Small extra move closer to the table")
            touch_down = pick_pos_joints.copy()
            touch_down[1] += EXTRA_DOWN_J2
            touch_down[2] += EXTRA_DOWN_J3
            touch_down[3] += EXTRA_DOWN_J4
            arm_move(touch_down, 700)
            time.sleep(0.5)
            current_pos = touch_down
        else:
            current_pos = pick_pos_joints

        print("[PICK] Step 4: Close gripper to grab the tray")
        gripper_close()
        time.sleep(0.8)

        print("[PICK] Step 5: Lift tray up for safe travel")
        lift_pos = current_pos.copy()
        lift_pos[2] += TRANSPORT_LIFT_J3
        arm_move(lift_pos, 800)
        time.sleep(0.5)

        print("[PICK] Step 6: Move over the drop position")
        drop_joints = drop_position[:5]
        arm_move(drop_joints, 800)
        time.sleep(0.5)

        print("[PICK] Step 6b: Lower down to the drop height")
        final_drop = drop_joints.copy()
        final_drop[2] += DROP_DOWN_J3
        arm_move(final_drop, 500)
        time.sleep(0.3)

        print("[PICK] Step 7: Open gripper to release the tray")
        gripper_open()
        time.sleep(0.5)

        print("[PICK] Step 8: Lift slightly so we don’t bump anything")
        clear_pos = final_drop.copy()
        clear_pos[2] += 15
        arm_move(clear_pos, 600)
        time.sleep(0.5)

        print("[PICK] Step 9: Go back to safe height")
        arm_move(safe_pos, 800)

        print(f"[PICK] Successfully picked from {position_name}")
        return True

    except Exception as e:
        print(f"[PICK] Failed to pick from {position_name}: {e}")
        try:
            arm_move_up()
            gripper_open()
        except:
            pass
        return False

# RETURN TRAYS TO ORIGINAL P_PICK POSITIONS
def return_trays(tray_moves):
    if not tray_moves:
        print("\n[TRAYS] No trays to return.")
        return

    print("\n" + "=" * 60)
    print("RETURNING TRAYS TO THEIR ORIGINAL POSITIONS")
    print("=" * 60)

    for i, move_info in enumerate(tray_moves, start=1):
        pos_name = move_info["position"]
        pick_pos = move_info["pick_pos"]
        drop_pos = move_info["drop_pos"]

        print(f"\n[TRAY] Returning tray {i}/{len(tray_moves)} back to {pos_name}")

        # 1) Go to a high safe pose before moving over the centre
        safe_pos = P_ABOVE_PILL[:5]
        arm_move(safe_pos, 800)
        time.sleep(0.5)

        # 2) Move above tray at center (also high)
        center_transport = drop_pos[:5].copy()
        center_transport[2] += TRANSPORT_LIFT_J3
        print("[TRAY] Moving above tray at center drop area")
        arm_move(center_transport, 800)
        time.sleep(0.5)

        # 3) Lower to grip the tray
        print("[TRAY] Lowering to grip the tray")
        center_grip = drop_pos[:5].copy()
        center_grip[2] += DROP_DOWN_J3
        arm_move(center_grip, 700)
        time.sleep(0.3)

        print("[TRAY] Closing gripper to hold the tray")
        gripper_close()
        time.sleep(0.5)

        # 4) LIFT HIGH before travelling back (this is the important part)
        print("[TRAY] Lifting tray for travel (high to avoid pills)")
        lift_from_center = center_grip.copy()
        lift_from_center[2] += RETURN_PICK_DOWN_J3   # now 50, not 0
        arm_move(lift_from_center, 800)
        time.sleep(0.5)

        # 5) OPTIONAL extra safety: go via global safe pose
        arm_move(safe_pos, 800)
        time.sleep(0.5)

        # 6) Move above original position (also high)
        print(f"[TRAY] Moving over original position: {pos_name}")
        pick_transport = pick_pos[:5].copy()
        pick_transport[2] += TRANSPORT_LIFT_J3
        arm_move(pick_transport, 800)
        time.sleep(0.5)

        # 7) Lower tray back to its original place
        print("[TRAY] Lowering tray back to its original place")
        arm_move(pick_pos[:5], 700)
        time.sleep(0.3)

        # 8) Release tray
        print("[TRAY] Releasing tray")
        gripper_open()
        time.sleep(0.5)

        # 9) Small lift to clear the tray
        clear_pick = pick_pos[:5].copy()
        clear_pick[2] += 15
        arm_move(clear_pick, 700)
        time.sleep(0.5)

    # Finish in a high safe pose
    arm_move(P_ABOVE_PILL[:5], 800)
    print("\n[TRAYS]  All trays returned to where they started.")

# PATIENT DATABASE WITH TIME SLOTS
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
                print(f"[INFO] Loaded {len(self.patients)} patients from JSON")
            except:
                print("[WARNING] Could not load patients file, switching to defaults.")
                self.create_default_patients()
        else:
            self.create_default_patients()
            self.save_patients()

        self.load_face_encodings()

    def create_default_patients(self):
        # Time-based schedule per patient
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
                        print(f"[INFO] Loaded face data for {data['full_name']}")
                except:
                    print(f"[WARNING] Could not load face data for {patient_id}")

    def save_patients(self):
        with open("/home/pi/Desktop/patients.json", 'w') as f:
            json.dump(self.patients, f, indent=2)

    def record_dispense(self, patient_id, slot_key):
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if patient_id in self.patients:
            self.patients[patient_id].setdefault("last_dispensed", {})
            self.patients[patient_id]["last_dispensed"][slot_key] = now_str
            self.save_patients()
            print(f"[INFO] Logged dispense for {patient_id} at {slot_key} -> {now_str}")

patient_db = PatientDatabase()

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
            cv2.putText(img, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return img, detections

# NAME NORMALIZATION & MAPPING
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

# --- SCAN + IMMEDIATE PICK FOR SLOT ---
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

# ROBOT MAIN LOGIC (GUI MODE, NO TERMINAL INPUT)
def robot_main(time_slot_key):
    global next_patient_decision

    print("[INFO] Medical Assistant DOFBOT started in GUI mode.")
    print(f"[INFO] Selected time slot: {time_slot_key}")
    print("[INFO] Waiting for a known patient in front of the camera...")

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
                    print(f"[FACE] All meds: {', '.join(patient['medications'])}")
                    print(f"[FACE] Meds for this slot ({time_slot_key}): "
                          f"{', '.join(patient.get('schedule', {}).get(time_slot_key, patient['medications']))}")

                elif (current_patient_id == pid and
                      patient_detection_time and
                      time.time() - patient_detection_time >= detection_confidence_time):

                    print("\n" + "=" * 60)
                    print(f"✅ PATIENT CONFIRMED: {patient['name']} for {time_slot_key}")
                    print("=" * 60)

                    picked_count, tray_moves = scan_and_pick_for_slot(patient, time_slot_key)

                    if picked_count > 0:
                        patient_db.record_dispense(patient['id'], time_slot_key)
                    else:
                        print("\n❌ No trays with pills were picked.")

                    print("\n[ACTION] Moving to a high, safe pose above the trays...")
                    arm_move(P_ABOVE_PILL[:5], 1000)
                    gripper_open()

                    # --- GUI: wait for user to empty trays and put them back ---
                    print("\n[TRAYS] Please remove the pills from the trays,")
                    print("        then place the EMPTY trays back at the center positions.")
                    print("        When ready, click 'Trays back in place' in the GUI.")

                    trays_ready_event.clear()
                    if gui_instance:
                        gui_instance.set_phase("WAIT_TRAYS")

                    while not trays_ready_event.is_set():
                        time.sleep(0.1)

                    if gui_instance:
                        gui_instance.set_phase("RETURNING_TRAYS")

                    return_trays(tray_moves)

                    print("\n[ACTION] Returning arm to home position...")
                    arm_move(JOINTS_HOME[:5], 1000)
                    gripper_open()

                    # --- GUI: ask if we should continue with next patient ---
                    print("\n[NEXT] Decide on the GUI if you want to scan for the next patient or stop.")
                    next_patient_event.clear()
                    next_patient_decision = "stop"
                    if gui_instance:
                        gui_instance.set_phase("WAIT_NEXT")

                    while not next_patient_event.is_set():
                        time.sleep(0.1)

                    decision = next_patient_decision
                    if gui_instance:
                        gui_instance.set_phase("IDLE")

                    if decision == "stop":
                        print("[INFO] Session ended by user from the GUI.")
                        break
                    else:
                        print("\n[READY] Resetting and waiting for the next patient...")
                        current_patient_id = None
                        patient_detection_time = None
                        time.sleep(2)
                        continue

            else:
                current_patient_id = None
                patient_detection_time = None

            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\n[INFO] Program interrupted from keyboard.")
    finally:
        arm_move(JOINTS_HOME[:5], 1000)
        gripper_open()
        print("[INFO] Robot logic finished for this run.")

# GUI WITH customtkinter
class DofbotGUI(ctk.CTk):
    def __init__(self): 
        super().__init__()

        global log_callback, gui_instance
        log_callback = self.append_log
        gui_instance = self

        self.title("DOCBOT Smart Pill Assistant")
        self.geometry("1200x700")

        # Robotic style: dark background + teal / cyan accents
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("dark-blue")
        self.configure(fg_color="#050816")  # deep navy / space blue

        self.time_slot_var = ctk.StringVar(value="09:00")

        #  TOP BAR
        top_bar = ctk.CTkFrame(self, fg_color="#070b1f", corner_radius=0)
        top_bar.pack(side="top", fill="x")

        title_label = ctk.CTkLabel(
            top_bar,
            text="🤖 DOFBOT Smart Pill Assistant",
            font=ctk.CTkFont(size=22, weight="bold"),
            text_color="#E5FEFF",
        )
        title_label.pack(side="left", padx=20, pady=10)

        # Time slot selector chip
        slot_chip = ctk.CTkFrame(top_bar, fg_color="#0f172a", corner_radius=18)
        slot_chip.pack(side="left", padx=20, pady=8)

        slot_label = ctk.CTkLabel(
            slot_chip,
            text="Time slot",
            font=ctk.CTkFont(size=13),
            text_color="#9CA3AF",
        )
        slot_label.pack(side="left", padx=(12, 4), pady=6)

        self.slot_menu = ctk.CTkOptionMenu(
            slot_chip,
            variable=self.time_slot_var,
            values=["09:00", "12:00", "14:00"],
            fg_color="#020617",
            button_color="#06B6D4",
            button_hover_color="#0891B2",
            dropdown_fg_color="#020617",
            text_color="#E5FEFF",
        )
        self.slot_menu.pack(side="left", padx=(4, 10), pady=6)

        # Start button
        self.start_button = ctk.CTkButton(
            top_bar,
            text="Start session",
            fg_color="#06B6D4",
            hover_color="#0891B2",
            font=ctk.CTkFont(size=14, weight="bold"),
            command=self.start_robot_clicked,
        )
        self.start_button.pack(side="right", padx=20, pady=8)

        # Exit button
        self.exit_button = ctk.CTkButton(
            top_bar,
            text="Exit",
            fg_color="#991B1B",
            hover_color="#7F1D1D",
            font=ctk.CTkFont(size=14, weight="bold"),
            command=self.close_app,
        )
        self.exit_button.pack(side="right", padx=10, pady=8)

        #  MAIN BODY
        main_frame = ctk.CTkFrame(self, fg_color="#020617")
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Left: camera + status
        left_panel = ctk.CTkFrame(main_frame, fg_color="#020617")
        left_panel.pack(side="left", fill="both", expand=True, padx=(5, 8), pady=5)

        # Camera frame with a subtle border
        cam_frame = ctk.CTkFrame(left_panel, fg_color="#020617", border_width=1, border_color="#1E293B")
        cam_frame.pack(fill="both", expand=True, padx=5, pady=(5, 10))

        self.camera_label = ctk.CTkLabel(
            cam_frame,
            text="Camera preview",
            text_color="#6B7280",
            font=ctk.CTkFont(size=14, slant="italic"),
        )
        self.camera_label.pack(fill="both", expand=True, padx=10, pady=10)

        # Status strip under the camera
        status_strip = ctk.CTkFrame(left_panel, fg_color="#020617")
        status_strip.pack(fill="x", padx=5, pady=(0, 5))

        self.phase_label = ctk.CTkLabel(
            status_strip,
            text="Status: idle (listening for 'hello dofbot')",
            font=ctk.CTkFont(size=13),
            text_color="#A5B4FC",
        )
        self.phase_label.pack(side="left", padx=8, pady=4)

        # Right: logs + interaction buttons
        right_panel = ctk.CTkFrame(main_frame, fg_color="#020617")
        right_panel.pack(side="right", fill="both", expand=True, padx=(8, 5), pady=5)

        # Logs box
        log_frame = ctk.CTkFrame(right_panel, fg_color="#020617")
        log_frame.pack(fill="both", expand=True, padx=5, pady=(5, 8))

        log_label = ctk.CTkLabel(
            log_frame,
            text="Robot console",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color="#E5E7EB",
        )
        log_label.pack(anchor="w", padx=6, pady=(6, 2))

        self.log_text = ctk.CTkTextbox(
            log_frame,
            wrap="word",
            fg_color="#020617",
            border_width=1,
            border_color="#1F2937",
            text_color="#E5E7EB",
        )
        self.log_text.pack(fill="both", expand=True, padx=6, pady=(0, 6))
        self.log_text.insert("end", "[INFO] GUI ready. I am listening for 'hello dofbot', or click 'Start session'.\n")
        self.log_text.configure(state="disabled")

        # Interaction buttons for tray + next patient
        ui_buttons = ctk.CTkFrame(right_panel, fg_color="#020617")
        ui_buttons.pack(fill="x", padx=5, pady=(0, 5))

        self.trays_button = ctk.CTkButton(
            ui_buttons,
            text=" Trays back in place",
            fg_color="#22C55E",
            hover_color="#16A34A",
            state="disabled",
            command=self.trays_ready_clicked,
        )
        self.trays_button.pack(side="left", padx=6, pady=6)

        self.next_patient_button = ctk.CTkButton(
            ui_buttons,
            text="Next patient",
            fg_color="#2563EB",
            hover_color="#1D4ED8",
            state="disabled",
            command=self.next_patient_clicked,
        )
        self.next_patient_button.pack(side="left", padx=6, pady=6)

        self.stop_session_button = ctk.CTkButton(
            ui_buttons,
            text="Stop session",
            fg_color="#DC2626",
            hover_color="#B91C1C",
            state="disabled",
            command=self.stop_session_clicked,
        )
        self.stop_session_button.pack(side="left", padx=6, pady=6)

        self.camera_thread_started = False
        self.robot_thread_running = False

        # Voice trigger state
        self.voice_stop_event = threading.Event()

        # Start background voice listener as soon as GUI loads
        voice_thread = threading.Thread(
            target=self._voice_listener_thread,
            daemon=True
        )
        voice_thread.start()

        # Camera polling
        self.after(30, self.update_camera_frame)

        # Handle window close
        self.protocol("WM_DELETE_WINDOW", self.close_app)

    #  status / phase management
    def set_phase(self, phase: str):
        """
        Update small status line and which buttons are active.
        phase can be: IDLE, WAIT_TRAYS, RETURNING_TRAYS, WAIT_NEXT
        """
        phase = phase.upper()
        if phase == "WAIT_TRAYS":
            self.phase_label.configure(text="Status: waiting for you to put trays back ✋")
            self.trays_button.configure(state="normal")
            self.next_patient_button.configure(state="disabled")
            self.stop_session_button.configure(state="disabled")
        elif phase == "RETURNING_TRAYS":
            self.phase_label.configure(text="Status: robot is returning trays ♻️")
            self.trays_button.configure(state="disabled")
            self.next_patient_button.configure(state="disabled")
            self.stop_session_button.configure(state="disabled")
        elif phase == "WAIT_NEXT":
            self.phase_label.configure(text="Status: choose next step (next patient / stop)")
            self.trays_button.configure(state="disabled")
            self.next_patient_button.configure(state="normal")
            self.stop_session_button.configure(state="normal")
        else:
            # IDLE or anything else
            self.phase_label.configure(text="Status: idle (listening for 'hello dofbot')")
            self.trays_button.configure(state="disabled")
            self.next_patient_button.configure(state="disabled")
            self.stop_session_button.configure(state="disabled")

    #  logging
    def append_log(self, line: str):
        self.log_text.configure(state="normal")
        self.log_text.insert("end", line + "\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    #  camera handling
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
                img = img.resize((580, 420))
                imgtk = ImageTk.PhotoImage(image=img)
                self.camera_label.configure(image=imgtk, text="")
                self.camera_label.image = imgtk
        self.after(30, self.update_camera_frame)

    #  background voice listener
    def _voice_listener_thread(self):
        """
        Background thread that waits for 'hello dofbot' (or 'hello' / 'hi dofbot').
        When detected, it starts the robot session if not already running.
        """
        while not self.voice_stop_event.is_set():
            print("\n[VOICE] Say 'hello dofbot' to start the robot session...")
            try:
                with sr.Microphone(device_index=MIC_INDEX) as mic:
                    recognizer.adjust_for_ambient_noise(mic, duration=0.5)
                    print("[VOICE] Listening in background...")

                    audio = recognizer.listen(mic, timeout=5, phrase_time_limit=5)
                    text = recognizer.recognize_google(audio).lower()
                    

                    if "hello dofbot" in text or "hi dofbot" in text or "hello" in text:
                        print("[VOICE] 👋 Wake word detected from background listener!")
                        # Start robot on main (GUI) thread
                        self.after(0, self.start_robot_via_voice)
                        return  # stop this listener thread

            except sr.WaitTimeoutError:
                # No speech – just loop again unless stop event is set
                continue
            except sr.UnknownValueError:
                print("[VOICE] Could not understand, listening again...")
                continue
            except sr.RequestError as e:
                print(f"[VOICE] Speech recognition service error: {e}")
                # Fall back: stop background listener, user can use Start button
                return
            except Exception as e:
                print(f"[VOICE] Microphone error in background listener: {e}")
                return

    def start_robot_via_voice(self):
        """
        Called from background voice listener when 'hello dofbot' is detected.
        Starts the robot session if not already running.
        """
        if self.robot_thread_running:
            print("[VOICE] Robot already running, ignoring wake word.")
            return

        self.start_camera()

        slot = self.time_slot_var.get()
        print(f"[GUI] Voice trigger: starting robot for time slot {slot}")
        self.robot_thread_running = True
        self.start_button.configure(state="disabled", text="Running...")

        # Stop listening during this session
        self.voice_stop_event.set()

        t = threading.Thread(target=self._run_robot_thread, args=(slot,), daemon=True)
        t.start()

    #  GUI buttons that talk to robot
    def trays_ready_clicked(self):
        """User confirms trays are empty and back at center positions."""
        trays_ready_event.set()
        self.append_log("[UI] You confirmed that trays are empty and back in place.")
        self.trays_button.configure(state="disabled")

    def next_patient_clicked(self):
        """User wants to continue with another patient."""
        global next_patient_decision
        next_patient_decision = "next"
        next_patient_event.set()
        self.append_log("[UI] You chose to continue with the next patient.")
        self.next_patient_button.configure(state="disabled")
        self.stop_session_button.configure(state="disabled")

    def stop_session_clicked(self):
        """User wants to stop the robot session."""
        global next_patient_decision
        next_patient_decision = "stop"
        next_patient_event.set()
        self.append_log("[UI] You chose to stop the session.")
        self.next_patient_button.configure(state="disabled")
        self.stop_session_button.configure(state="disabled")

    #  robot start via button
    def start_robot_clicked(self):
        if self.robot_thread_running:
            print("[INFO] Robot is already running.")
            return

        self.start_camera()

        slot = self.time_slot_var.get()
        print(f"[GUI] Start button: starting robot for time slot {slot}")
        self.robot_thread_running = True
        self.start_button.configure(state="disabled", text="Running...")

        # Stop voice listener during the active session
        self.voice_stop_event.set()

        t = threading.Thread(target=self._run_robot_thread, args=(slot,), daemon=True)
        t.start()

    def _run_robot_thread(self, slot_key):
        try:
            self.set_phase("IDLE")
            self.append_log("[INFO] Starting robot logic...")
            robot_main(slot_key)
        finally:
            self.robot_thread_running = False
            print("[GUI] Robot thread finished.")
            self.after(0, self.on_robot_finished)

    def on_robot_finished(self):
        self.start_button.configure(state="normal", text="Start session")
        self.set_phase("IDLE")
        self.append_log("[INFO] Robot session ended. You can change the time slot and start again.")

        # Re-enable voice listening for the next session
        self.voice_stop_event.clear()
        voice_thread = threading.Thread(
            target=self._voice_listener_thread,
            daemon=True
        )
        voice_thread.start()

    #  close app
    def close_app(self):
        """Stop camera, close GUI and exit."""
        self.stop_camera()
        try:
            self.voice_stop_event.set()
        except Exception:
            pass
        try:
            self.destroy()
        except:
            pass
        os._exit(0)

# ENTRY POINT

if __name__ == "__main__":
    app = DofbotGUI()
    app.mainloop()