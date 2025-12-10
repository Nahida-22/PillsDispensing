#!/usr/bin/env python3
# coding=utf-8

import time
from Arm_Lib import Arm_Device


class ArmController:
    """
    Handles all low-level DOFBOT movement:
    - joint positions
    - gripper open/close
    - picking trays/pills
    - returning trays to their original positions
    """

    def __init__(self):
        self.arm = Arm_Device()
        time.sleep(0.1)
        # Home / neutral pose
        self.JOINTS_HOME = [90, 135, 20, 25, 90, 30]
        self.JOINTS_SCAN_HEIGHT = [90, 100, 0, 40, 90, 30]
        # Pill inspection positions (camera looking at trays)

        # Right side
        self.P_PILL_1 = [10, 138, -35, 14, 269, 30]   # Right 1
        self.P_PILL_2 = [40, 138, -35, 14, 269, 30]   # Right 2
        self.P_PILL_3 = [70, 142, -35, 18, 269, 30]   # Right 3

        # Left side
        self.P_PILL_4 = [174, 138, -35, 14, 269, 30]  # Left 1
        self.P_PILL_5 = [140, 138, -35, 14, 269, 30]  # Left 2
        self.P_PILL_6 = [110, 142, -35, 18, 269, 30]  # Left 3

        # Pill picking positions (slightly lower than the inspection positions)
        # Right
        self.P_PICK_1 = [10, 66, 15, 33, 270, 30]     # Right 1
        self.P_PICK_2 = [44, 66, 20, 28, 270, 30]     # Right 2
        self.P_PICK_3 = [70, 66, 15, 33, 270, 30]     # Right 3

        # Left
        self.P_PICK_4 = [170, 66, 15, 33, 270, 30]    # Left 1
        self.P_PICK_5 = [140, 66, 20, 28, 270, 30]    # Left 2
        self.P_PICK_6 = [110, 66, 15, 33, 270, 30]    # Left 3

        # Center placement positions (where the trays are dropped)
        self.P_CENTER_PLACE   = [90, 66, 20, 29, 270, 30]  # First slot
        self.P_CENTER_PLACE_2 = [90, 88, 20, 29, 270, 30]  # Second slot

        # Safe position above table to move around without hitting anything
        self.P_ABOVE_PILL = [90, 80, 50, 50, 270, 30]

        # GRIPPER CONFIG
        self.GRIPPER_OPEN_ANGLE = 60
        self.GRIPPER_CLOSE_ANGLE = 145  # tighter grip

        # LIFT & DROP TUNING
        self.TRANSPORT_LIFT_J3 = 50   # how much we lift at joint 3 when travelling
        self.DROP_DOWN_J3 = 0        # small lowering for drop
        self.RETURN_PICK_DOWN_J3 = 0 # small offset when lifting tray during return

        # Move to home on startup
        self.move_home()

    def move_home(self):
        self.arm.Arm_serial_servo_write6_array(self.JOINTS_HOME, 1500)
        time.sleep(2)

    def move_joints(self, p, s_time=500):
        """Send angles to servos 1–5 with a small stagger for servo 5."""
        for i in range(5):
            sid = i + 1
            if sid == 5:
                time.sleep(.1)
                self.arm.Arm_serial_servo_write(sid, p[i], int(s_time * 1.2))
            else:
                self.arm.Arm_serial_servo_write(sid, p[i], s_time)
            time.sleep(.01)
        time.sleep(s_time / 1000)

    def move_up(self):
        """Bring the arm up to a neutral pose (used for recovery)."""
        self.arm.Arm_serial_servo_write(2, 90, 1500)
        self.arm.Arm_serial_servo_write(3, 90, 1500)
        self.arm.Arm_serial_servo_write(4, 90, 1500)
        time.sleep(0.1)

    #gripper

    def _clamp_block(self, enable):
        if enable == 0:
            self.arm.Arm_serial_servo_write(6, self.GRIPPER_OPEN_ANGLE, 400)  # open
        else:
            self.arm.Arm_serial_servo_write(6, self.GRIPPER_CLOSE_ANGLE, 400) # close
        time.sleep(.5)

    def open_gripper(self):
        self._clamp_block(0)

    def close_gripper(self):
        self._clamp_block(1)

    #tray/pill pick

    def pick_tray(self, position_name, pick_position, drop_position):
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
            self.open_gripper()
            time.sleep(0.5)

            print("[PICK] Step 2: Move to safe height above the table")
            safe_pos = self.P_ABOVE_PILL[:5]
            self.move_joints(safe_pos, 800)
            time.sleep(0.5)

            print("[PICK] Step 3: Move down to tray")
            pick_pos_joints = pick_position[:5]
            self.move_joints(pick_pos_joints, 1000)
            time.sleep(0.5)

            if any([EXTRA_DOWN_J2, EXTRA_DOWN_J3, EXTRA_DOWN_J4]):
                print("[PICK] Step 3b: Small extra move closer to the table")
                touch_down = pick_pos_joints.copy()
                touch_down[1] += EXTRA_DOWN_J2
                touch_down[2] += EXTRA_DOWN_J3
                touch_down[3] += EXTRA_DOWN_J4
                self.move_joints(touch_down, 700)
                time.sleep(0.5)
                current_pos = touch_down
            else:
                current_pos = pick_pos_joints

            print("[PICK] Step 4: Close gripper to grab the tray")
            self.close_gripper()
            time.sleep(0.8)

            print("[PICK] Step 5: Lift tray up for safe travel")
            lift_pos = current_pos.copy()
            lift_pos[2] += self.TRANSPORT_LIFT_J3
            self.move_joints(lift_pos, 800)
            time.sleep(0.5)

            print("[PICK] Step 6: Move over the drop position")
            drop_joints = drop_position[:5]
            self.move_joints(drop_joints, 800)
            time.sleep(0.5)

            print("[PICK] Step 6b: Lower down to the drop height")
            final_drop = drop_joints.copy()
            final_drop[2] += self.DROP_DOWN_J3
            self.move_joints(final_drop, 500)
            time.sleep(0.3)

            print("[PICK] Step 7: Open gripper to release the tray")
            self.open_gripper()
            time.sleep(0.5)

            print("[PICK] Step 8: Lift slightly so we don’t bump anything")
            clear_pos = final_drop.copy()
            clear_pos[2] += 15
            self.move_joints(clear_pos, 600)
            time.sleep(0.5)

            print("[PICK] Step 9: Go back to safe height")
            self.move_joints(safe_pos, 800)

            print(f"[PICK] Successfully picked from {position_name}")
            return True

        except Exception as e:
            print(f"[PICK] Failed to pick from {position_name}: {e}")
            try:
                self.move_up()
                self.open_gripper()
            except:
                pass
            return False

    #return trays

    def return_trays(self, tray_moves):
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

            safe_pos = self.P_ABOVE_PILL[:5]
            self.move_joints(safe_pos, 800)
            time.sleep(0.5)

            center_transport = drop_pos[:5].copy()
            center_transport[2] += self.TRANSPORT_LIFT_J3
            print("[TRAY] Moving above tray at center drop area")
            self.move_joints(center_transport, 800)
            time.sleep(0.5)

            print("[TRAY] Lowering to grip the tray")
            center_grip = drop_pos[:5].copy()
            center_grip[2] += self.DROP_DOWN_J3
            self.move_joints(center_grip, 700)
            time.sleep(0.3)

            print("[TRAY] Closing gripper to hold the tray")
            self.close_gripper()
            time.sleep(0.5)

            print("[TRAY] Lifting tray for travel")
            lift_from_center = center_grip.copy()
            lift_from_center[2] += self.RETURN_PICK_DOWN_J3
            self.move_joints(lift_from_center, 800)
            time.sleep(0.5)

            print(f"[TRAY] Moving over original position: {pos_name}")
            pick_transport = pick_pos[:5].copy()
            pick_transport[2] += self.TRANSPORT_LIFT_J3
            self.move_joints(pick_transport, 800)
            time.sleep(0.5)

            print("[TRAY] Lowering tray back to its original place")
            self.move_joints(pick_pos[:5], 700)
            time.sleep(0.3)

            print("[TRAY] Releasing tray")
            self.open_gripper()
            time.sleep(0.5)

            clear_pick = pick_pos[:5].copy()
            clear_pick[2] += 15
            self.move_joints(clear_pick, 700)
            time.sleep(0.5)

        self.move_joints(self.P_ABOVE_PILL[:5], 800)
        print("\n[TRAYS]  All trays returned to where they started.")