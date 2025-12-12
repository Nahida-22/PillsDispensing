# robot_logic.py
#!/usr/bin/env python3
# coding=utf-8

import time
import threading
import cv2

from arm_control import arm_move, gripper_open, JOINTS_HOME, P_ABOVE_PILL
from patient_db import patient_db
from perception import detect_patient, scan_and_pick_for_slot, get_latest_frame
from arm_control import return_trays

#  SHARED STATE FOR GUI <-> ROBOT 
trays_ready_event = threading.Event()
next_patient_event = threading.Event()
next_patient_decision = "stop"  # "next" or "stop"
gui_instance = None


def set_gui_instance(gui):
    """GUI calls this once so robot logic can update GUI phase labels."""
    global gui_instance
    gui_instance = gui


def robot_main(time_slot_key: str):
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
                    print(
                        f"[FACE] Meds for this slot ({time_slot_key}): "
                        f"{', '.join(patient.get('schedule', {}).get(time_slot_key, patient['medications']))}"
                    )

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
                        print("\nNo trays with pills were picked.")

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