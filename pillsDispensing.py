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
P_CENTER_PLACE_2 = [90, 88, 20, 29, 270, 30]  # Second slot (just next to it)

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

# LIFT & DROP TUNINGa
TRANSPORT_LIFT_J3 = 50   # how much we lift at joint 3 when travelling
DROP_DOWN_J3= 0  # how much we lower at drop so the tray doesn’t fall from too high
RETURN_PICK_DOWN_J3 = 0  # how much we lower when returning trays to original spots

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

        safe_pos = P_ABOVE_PILL[:5]
        arm_move(safe_pos, 800)
        time.sleep(0.5)

        center_transport = drop_pos[:5].copy()
        center_transport[2] += TRANSPORT_LIFT_J3
        print("[TRAY] Moving above tray at center drop area")
        arm_move(center_transport, 800)
        time.sleep(0.5)

        print("[TRAY] Lowering to grip the tray")
        center_grip = drop_pos[:5].copy()
        center_grip[2] += DROP_DOWN_J3
        arm_move(center_grip, 700)
        time.sleep(0.3)

        print("[TRAY] Closing gripper to hold the tray")
        gripper_close()
        time.sleep(0.5)

        print("[TRAY] Lifting tray for travel")
        lift_from_center = center_grip.copy()
        lift_from_center[2] += RETURN_PICK_DOWN_J3
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

# SCAN PILL POSITIONS
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
        print(f"\n CHECKING {position_name}")
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

        arm_move(P_ABOVE_PILL[:5], 1000)
        time.sleep(0.3)

    return found_pills

# PICK PILLS PER TIME SLOT
def pick_all_detected_pills(found_pills, patient, slot_key):
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
            drop_position = P_CENTER_PLACE
        elif med_index == 1:
            drop_position = P_CENTER_PLACE_2
        else:
            drop_position = P_CENTER_PLACE_2

        print(f"[PICK] -> Using tray {pos_name} with meds: {', '.join(meds_for_pos)}")
        print(f"[PICK] -> Dropping at: {'P_CENTER_PLACE' if med_index == 0 else 'P_CENTER_PLACE_2'}")

        success = pick_pill_simple(pos_name, pick_pos, drop_position)
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

                    found_pills = scan_pill_positions(patient['name'])

                    if found_pills:
                        print(f"\n✅ Found trays at {len(found_pills)} positions.")
                        picked_count, tray_moves = pick_all_detected_pills(found_pills, patient, time_slot_key)
                        if picked_count > 0:
                            patient_db.record_dispense(patient['id'], time_slot_key)
                    else:
                        print("\n❌ No trays with pills were detected.")
                        picked_count, tray_moves = 0, []

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

                    # Robot waits here until GUI button is pressed
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

        self.title("DOFBOT Pill Assistant")
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
            text="Status: idle",
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
        self.log_text.insert("end", "[INFO] GUI ready. Choose a time slot and click 'Start session'.\n")
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
            self.phase_label.configure(text="Status: idle")
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

    #  robot start 
    def start_robot_clicked(self):
        if self.robot_thread_running:
            print("[INFO] Robot is already running.")
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
            self.set_phase("IDLE")
            robot_main(slot_key)
        finally:
            self.robot_thread_running = False
            print("[GUI] Robot thread finished.")
            self.after(0, self.on_robot_finished)

    def on_robot_finished(self):
        self.start_button.configure(state="normal", text="Start session")
        self.set_phase("IDLE")
        self.append_log("[INFO] Robot session ended. You can change the time slot and start again.")

    #  close app 
    def close_app(self):
        """Stop camera, close GUI and exit."""
        self.stop_camera()
        try:
            self.destroy()
        except:
            pass
        os._exit(0)

# ENTRY POINT
if __name__ == "__main__":
    app = DofbotGUI()
    app.mainloop()