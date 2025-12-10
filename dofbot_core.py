#!/usr/bin/env python3
# coding=utf-8

import cv2
import time
import threading
import builtins

from arm_control import ArmController
from patients_db import PatientDatabase
from vision_pills import PillVision

# GLOBALS FOR ROBOT / GUI LINK
latest_frame = None                 
log_callback = None                  

trays_ready_event = threading.Event()
next_patient_event = threading.Event()
next_patient_decision = "stop"       \

gui_instance = None         


def set_log_callback(cb):
    """GUI calls this so robot logs also appear in the GUI console."""
    global log_callback
    log_callback = cb


def set_gui_instance(gui):
    """GUI calls this so robot can update GUI state (phase labels, etc.)."""
    global gui_instance
    gui_instance = gui


def update_latest_frame(frame):
    """GUI camera loop calls this to provide the freshest frame to the robot."""
    global latest_frame
    latest_frame = frame


def get_latest_frame():
    """Robot logic uses this to read the most recent camera frame."""
    global latest_frame
    if latest_frame is None:
        return None
    return latest_frame.copy()


# Intercept print() so logs appear both in terminal and in GUI
_real_print = builtins.print
def gui_print(*args, **kwargs):
    s = " ".join(str(a) for a in args)
    _real_print(*args, **kwargs)
    if log_callback:
        log_callback(s)
builtins.print = gui_print

# MODULE SINGLETONS
arm = ArmControlle()
patient_db = PatientDatabase()
vision = PillVision(get_latest_frame, patient_db, arm)

# ROBOT MAIN LOGIC (GUI MODE)
def robot_main(time_slot_key):
    global next_patient_decision

    print("[INFO] Medical Assistant DOFBOT started in GUI mode.")
    print(f"[INFO] Selected time slot: {time_slot_key}")
    print("[INFO] Waiting for a known patient in front of the camera...")

    arm.open_gripper()

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

            display_frame, patients = vision.detect_patient(display_frame)

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
                    print(f"PATIENT CONFIRMED: {patient['name']} for {time_slot_key}")
                    print("=" * 60)

                    found_pills = vision.scan_pill_positions(patient['name'])

                    if found_pills:
                        print(f"\nFound trays at {len(found_pills)} positions.")
                        picked_count, tray_moves = vision.pick_all_detected_pills(
                            found_pills, patient, time_slot_key
                        )
                        if picked_count > 0:
                            patient_db.record_dispense(patient['id'], time_slot_key)
                    else:
                        print("\nNo trays with pills were detected.")
                        picked_count, tray_moves = 0, []

                    print("\n[ACTION] Moving to a high, safe pose above the trays...")
                    arm.move_joints(arm.P_ABOVE_PILL[:5], 1000)
                    arm.open_gripper()

                    # GUI: wait for user to empty trays and put them back
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

                    arm.return_trays(tray_moves)

                    print("\n[ACTION] Returning arm to home position...")
                    arm.move_joints(arm.JOINTS_HOME[:5], 1000)
                    arm.open_gripper()

                    # GUI: ask if we should continue with next patient
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
        arm.move_joints(arm.JOINTS_HOME[:5], 1000)
        arm.open_gripper()
        print("[INFO] Robot logic finished for this run.")


if __name__ == "__main__":
    print("This module is meant to be used from dofbot_gui.py, not run directly.")