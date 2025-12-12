# gui_app.py
#!/usr/bin/env python3
# coding=utf-8

import os
import threading
import builtins

import cv2
import customtkinter as ctk
from PIL import Image, ImageTk
import speech_recognition as sr

from perception import set_latest_frame
import robot_logic
from robot_logic import (
    robot_main,
    trays_ready_event,
    next_patient_event,
)

#  SPEECH RECOGNITION CONFIG 
recognizer = sr.Recognizer()
MIC_INDEX = 3  # adjust if needed for your Pi


#  GLOBALS FOR LOGGING 
log_callback = None

# Intercept print() so logs appear both in terminal and in GUI
_real_print = builtins.print


def gui_print(*args, **kwargs):
    s = " ".join(str(a) for a in args)
    _real_print(*args, **kwargs)
    global log_callback
    if log_callback:
        log_callback(s)


builtins.print = gui_print


class DofbotGUI(ctk.CTk):
    def __init__(self):
        super().__init__()

        global log_callback
        log_callback = self.append_log

        # Let robot logic know who the GUI is
        robot_logic.set_gui_instance(self)

        self.title("DOCBOT Smart Pill Assistant")
        self.geometry("1200x700")

        # Robotic style: dark background + teal / cyan accents
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("dark-blue")
        self.configure(fg_color="#050816")  # deep navy / space blue

        self.time_slot_var = ctk.StringVar(value="09:00")

        # Camera state
        self.camera_running = False
        self.cap = None

        # Robot state
        self.robot_thread_running = False

        # Voice trigger state
        self.voice_stop_event = threading.Event()

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

        cam_frame = ctk.CTkFrame(
            left_panel,
            fg_color="#020617",
            border_width=1,
            border_color="#1E293B"
        )
        cam_frame.pack(fill="both", expand=True, padx=5, pady=(5, 10))

        self.camera_label = ctk.CTkLabel(
            cam_frame,
            text="Camera preview",
            text_color="#6B7280",
            font=ctk.CTkFont(size=14, slant="italic"),
        )
        self.camera_label.pack(fill="both", expand=True, padx=10, pady=10)

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
        self.log_text.insert(
            "end",
            "[INFO] GUI ready. I am listening for 'hello dofbot', or click 'Start session'.\n"
        )
        self.log_text.configure(state="disabled")

        # Interaction buttons
        ui_buttons = ctk.CTkFrame(right_panel, fg_color="#020617")
        ui_buttons.pack(fill="x", padx=5, pady=(0, 5))

        from robot_logic import next_patient_decision 

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

        # Start background voice listener
        voice_thread = threading.Thread(
            target=self._voice_listener_thread,
            daemon=True
        )
        voice_thread.start()

        # Camera polling
        self.after(30, self.update_camera_frame)

        # Handle window close
        self.protocol("WM_DELETE_WINDOW", self.close_app)

    #  STATUS / PHASE MANAGEMENT 
    def set_phase(self, phase: str):
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
            self.phase_label.configure(text="Status: idle (listening for 'hello dofbot')")
            self.trays_button.configure(state="disabled")
            self.next_patient_button.configure(state="disabled")
            self.stop_session_button.configure(state="disabled")

    #  LOGGING 
    def append_log(self, line: str):
        self.log_text.configure(state="normal")
        self.log_text.insert("end", line + "\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    #  CAMERA HANDLING 
    def start_camera(self):
        if self.camera_running:
            return
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("[ERROR] Cannot open camera from GUI.")
            return
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.camera_running = True
        print("[INFO] Camera started from GUI.")

    def stop_camera(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.camera_running = False
        print("[INFO] Camera stopped and released.")

    def update_camera_frame(self):
        if self.camera_running and self.cap is not None:
            ret, frame = self.cap.read()
            if ret:
                # send to perception module
                set_latest_frame(frame)

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(rgb)
                img = img.resize((580, 420))
                imgtk = ImageTk.PhotoImage(image=img)
                self.camera_label.configure(image=imgtk, text="")
                self.camera_label.image = imgtk
        self.after(30, self.update_camera_frame)

    #  BACKGROUND VOICE LISTENER 
    def _voice_listener_thread(self):
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
                        self.after(0, self.start_robot_via_voice)
                        return  # stop listener

            except sr.WaitTimeoutError:
                continue
            except sr.UnknownValueError:
                print("[VOICE] Could not understand, listening again...")
                continue
            except sr.RequestError as e:
                print(f"[VOICE] Speech recognition service error: {e}")
                return
            except Exception as e:
                print(f"[VOICE] Microphone error in background listener: {e}")
                return

    def start_robot_via_voice(self):
        if self.robot_thread_running:
            print("[VOICE] Robot already running, ignoring wake word.")
            return

        self.start_camera()

        slot = self.time_slot_var.get()
        print(f"[GUI] Voice trigger: starting robot for time slot {slot}")
        self.robot_thread_running = True
        self.start_button.configure(state="disabled", text="Running...")

        self.voice_stop_event.set()

        t = threading.Thread(
            target=self._run_robot_thread,
            args=(slot,),
            daemon=True
        )
        t.start()

    #  GUI BUTTON CALLBACKS 
    def trays_ready_clicked(self):
        trays_ready_event.set()
        self.append_log("[UI] You confirmed that trays are empty and back in place.")
        self.trays_button.configure(state="disabled")

    def next_patient_clicked(self):
        robot_logic.next_patient_decision = "next"
        next_patient_event.set()
        self.append_log("[UI] You chose to continue with the next patient.")
        self.next_patient_button.configure(state="disabled")
        self.stop_session_button.configure(state="disabled")

    def stop_session_clicked(self):
        robot_logic.next_patient_decision = "stop"
        next_patient_event.set()
        self.append_log("[UI] You chose to stop the session.")
        self.next_patient_button.configure(state="disabled")
        self.stop_session_button.configure(state="disabled")

    #  START ROBOT VIA BUTTON 
    def start_robot_clicked(self):
        if self.robot_thread_running:
            print("[INFO] Robot is already running.")
            return

        self.start_camera()

        slot = self.time_slot_var.get()
        print(f"[GUI] Start button: starting robot for time slot {slot}")
        self.robot_thread_running = True
        self.start_button.configure(state="disabled", text="Running...")

        self.voice_stop_event.set()

        t = threading.Thread(
            target=self._run_robot_thread,
            args=(slot,),
            daemon=True
        )
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

    #  CLOSE APP 
    def close_app(self):
        self.stop_camera()
        try:
            self.voice_stop_event.set()
        except Exception:
            pass
        try:
            self.destroy()
        except Exception:
            pass
        os._exit(0)