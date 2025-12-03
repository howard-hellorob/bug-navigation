import time
import numpy as np
from mbot_bridge.api import MBot

GO_TO_GOAL = 0
AVOID_OBSTACLE = 1

def normalize_angle(a):
    return (a + np.pi) % (2*np.pi) - np.pi

def find_min_ray_in_slice(ranges, thetas, target_angle, slice_size):
    ranges = np.array(ranges)
    thetas = np.array(thetas)
    diffs = np.array([normalize_angle(th - target_angle) for th in thetas])
    mask = np.abs(diffs) <= slice_size/2
    r = ranges[mask]
    if r.size == 0:    # no valid rays in slice
        return np.inf
    r = r[r > 0]
    return np.min(r) if r.size > 0 else np.inf

robot = MBot()
robot.reset_odometry()

goal_x = float(input("Goal x [m]: "))
goal_y = float(input("Goal y [m]: "))
goal_th = np.deg2rad(float(input("Goal theta [deg]: ")))

Kp_dist, Kp_ang = 0.6, 1.5
v_max, w_max = 0.4, 1.0
slice_size = np.deg2rad(30)
obstacle_clearance = 0.45
pos_tol, ang_tol = 0.05, np.deg2rad(5)

state = GO_TO_GOAL
prev_state = None

try:
    while True:
        x, y, th = robot.read_odometry()
        dx, dy = goal_x - x, goal_y - y
        dist = np.hypot(dx, dy)
        goal_dir_world = np.arctan2(dy, dx)
        goal_dir_robot = normalize_angle(goal_dir_world - th)

        # goal check
        if dist < pos_tol and abs(normalize_angle(goal_th - th)) < ang_tol:
            print("AT_GOAL")
            robot.stop()
            break

        if state != prev_state:
            print("STATE:", "GO_TO_GOAL" if state == GO_TO_GOAL else "AVOID_OBSTACLE")
            prev_state = state

        ranges, thetas = robot.read_lidar()
        min_ray = find_min_ray_in_slice(ranges, thetas, goal_dir_robot, slice_size)

        if state == GO_TO_GOAL:
            if min_ray < obstacle_clearance:
                state = AVOID_OBSTACLE
                continue

            v = np.clip(Kp_dist * dist, -v_max, v_max)
            w = np.clip(Kp_ang * goal_dir_robot, -w_max, w_max)
            robot.drive(v, 0.0, w)

        else:  # AVOID_OBSTACLE
            #tells it to turn left while moving forward like in the video
            robot.drive(0.15, 0.0, 0.5)
            # if path is now clear, go back to GO_TO_GOAL
            if min_ray > obstacle_clearance:
                state = GO_TO_GOAL

        time.sleep(0.05)

finally:
    robot.stop()
