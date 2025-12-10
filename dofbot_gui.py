#!/usr/bin/env python3
# coding=utf-8
import cv2
import threading
import os
import customtkinter as ctk
from PIL import Image, ImageTk
import dofbot_core as core
# camera state (GUI side)
camera_running = False
cap = None

class DofbotGUI(ctk.CTk):
    def __init__(self):
        super().__init__()

        core.set_log_callback(self.append_log)
        core.set_gui_instance(self)

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
        global cap, camera_running

        if camera_running and cap is not None:
            ret, frame = cap.read()
            if ret:
                # send frame to the robot side
                core.update_latest_frame(frame)

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(rgb)
                img = img.resize((580, 420))
                imgtk = ImageTk.PhotoImage(image=img)
                self.camera_label.configure(image=imgtk, text="")
                self.camera_label.image = imgtk
        self.after(30, self.update_camera_frame)

    #  GUI buttons that talk to the robot 
    def trays_ready_clicked(self):
        """User confirms trays are empty and back at center positions."""
        core.trays_ready_event.set()
        self.append_log("[UI] You confirmed that trays are empty and back in place.")
        self.trays_button.configure(state="disabled")

    def next_patient_clicked(self):
        """User wants to continue with another patient."""
        core.next_patient_decision = "next"
        core.next_patient_event.set()
        self.append_log("[UI] You chose to continue with the next patient.")
        self.next_patient_button.configure(state="disabled")
        self.stop_session_button.configure(state="disabled")

    def stop_session_clicked(self):
        """User wants to stop the robot session."""
        core.next_patient_decision = "stop"
        core.next_patient_event.set()
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
            core.robot_main(slot_key)
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


if __name__ == "__main__":
    app = DofbotGUI()
    app.mainloop()