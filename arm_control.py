# arm_control.py
#!/usr/bin/env python3
# coding=utf-8

import time
from Arm_Lib import Arm_Device

#  DOFBOT INIT 
Arm = Arm_Device()
time.sleep(0.1)

#  ARM POSITIONS 
JOINTS_HOME = [90, 135, 20, 25, 90, 30]           # Home / neutral pose
JOINTS_SCAN_HEIGHT = [90, 100, 0, 40, 90, 30]     # Not used directly for now

# Pill inspection positions (camera looking at trays)
# Right side
P_PILL_1 = [10, 138, -35, 14, 269, 30]
P_PILL_2 = [45, 138, -35, 14, 269, 30]   
P_PILL_3 = [80, 142, -35, 18, 269, 30] 
# Left side
P_PILL_4 = [174, 138, -35, 14, 269, 30]  
P_PILL_5 = [140, 138, -35, 14, 269, 30]  
P_PILL_6 = [110, 142, -35, 18, 269, 30]  
# Pill picking positions (slightly lower than the inspection positions)
# Right
P_PICK_1 = [10, 66, 15, 33, 270, 30] 
P_PICK_2 = [44, 66, 20, 28, 270, 30]    
P_PICK_3 = [70, 66, 15, 33, 270, 30]    

# Left
P_PICK_4 = [170, 66, 15, 33, 270, 30] 
P_PICK_5 = [140, 66, 20, 28, 270, 30] 
P_PICK_6 = [110, 66, 15, 33, 270, 30]   

# Center placement positions (where the trays are dropped)
P_CENTER_PLACE   = [90, 66, 20, 29, 270, 30] 
P_CENTER_PLACE_2 = [90, 88, 20, 29, 270, 30]  

# Safe position above table to move around without hitting anything
P_ABOVE_PILL = [90, 80, 50, 50, 270, 30]

# Move to home on startup
Arm.Arm_serial_servo_write6_array(JOINTS_HOME, 1500)
time.sleep(2)

#  GRIPPER CONFIG 
GRIPPER_OPEN_ANGLE = 60
GRIPPER_CLOSE_ANGLE = 145  # increase if you want the clamp to squeeze harder

def arm_clamp_block(enable: int):
    """Low-level control of the gripper servo."""
    if enable == 0:
        Arm.Arm_serial_servo_write(6, GRIPPER_OPEN_ANGLE, 400)  # open
    else:
        Arm.Arm_serial_servo_write(6, GRIPPER_CLOSE_ANGLE, 400)  # close
    time.sleep(.5)

def gripper_open():
    arm_clamp_block(0)

def gripper_close():
    arm_clamp_block(1)

#  BASIC ARM MOVE HELPERS 
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

#  LIFT & DROP TUNING 
TRANSPORT_LIFT_J3 = 50   # how much we lift at joint 3 when travelling
DROP_DOWN_J3      = 0    # how much we lower at drop
RETURN_PICK_DOWN_J3 = 50 # extra lift when returning trays

#  SIMPLE PICK ROUTINE 
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
        except Exception:
            pass
        return False


#  RETURN TRAYS TO ORIGINAL P_PICK POSITIONS 
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

        # 4) LIFT HIGH before travelling back
        print("[TRAY] Lifting tray for travel (high to avoid pills)")
        lift_from_center = center_grip.copy()
        lift_from_center[2] += RETURN_PICK_DOWN_J3
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