# patient_db.py
#!/usr/bin/env python3
# coding=utf-8

import json
import os
from datetime import datetime

import face_recognition


#  NAME NORMALIZATION & MAPPING 
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


#  PATIENT DATABASE WITH TIME SLOTS 
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
            except Exception:
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
                except Exception:
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


# Single shared instance
patient_db = PatientDatabase()