#!/usr/bin/env python3
# coding=utf-8

import json
import os
from datetime import datetime
import face_recognition


class PatientDatabase:
    """
    Manages patient details, schedules and face encodings.
    """
    def __init__(self, json_path="/home/pi/Desktop/patients.json"):
        self.json_path = json_path
        self.patients = {}
        self.face_encodings = []
        self.patient_names = []
        self.load_patients()

    # JSON load/save
    def load_patients(self):
        if os.path.exists(self.json_path):
            try:
                with open(self.json_path, 'r') as f:
                    self.patients = json.load(f)
                print(f"[INFO] Loaded {len(self.patients)} patients from JSON")
            except Exception as e:
                print(f"[WARNING] Could not load patients file ({e}), using defaults.")
                self._create_default_patients()
        else:
            self._create_default_patients()
            self.save_patients()

        self._load_face_encodings()

    def save_patients(self):
        try:
            with open(self.json_path, 'w') as f:
                json.dump(self.patients, f, indent=2)
        except Exception as e:
            print(f"[WARNING] Could not save patients JSON: {e}")

    # default data
    def _create_default_patients(self):
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

    # faces
    def _load_face_encodings(self):
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
                except Exception as e:
                    print(f"[WARNING] Could not load face data for {patient_id}: {e}")

    # logging dispense
    def record_dispense(self, patient_id, slot_key):
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if patient_id in self.patients:
            self.patients[patient_id].setdefault("last_dispensed", {})
            self.patients[patient_id]["last_dispensed"][slot_key] = now_str
            self.save_patients()
            print(f"[INFO] Logged dispense for {patient_id} at {slot_key} -> {now_str}")