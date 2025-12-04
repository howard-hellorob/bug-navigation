import time
import numpy as np
from mbot_bridge.api import MBot



def normalize_angle(angle):
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle < -np.pi:
        angle += 2 * np.pi
    return angle

def find_min_ray_in_slice(ranges, thetas, target_angle, slice_size):
    min_dist = float('inf')
    half_slice = slice_size / 2.0
    
    for r, t in zip(ranges, thetas):
        if r == 0: continue
        diff = normalize_angle(t - target_angle)
        if abs(diff) < half_slice:
            if r < min_dist:
                min_dist = r
    
    return min_dist if min_dist != float('inf') else 10.0

def find_min_dist_all(ranges, thetas):
    min_dist = float('inf')
    min_angle = 0
    for r, t in zip(ranges, thetas):
        if r == 0: continue
        if r < min_dist:
            min_dist = r
            min_angle = t
    return min_dist, min_angle

def cross_product(v1, v2):
    res = np.zeros(3)
    res[0] = v1[1]*v2[2] - v1[2]*v2[1]
    res[1] = v1[2]*v2[0] - v1[0]*v2[2]
    res[2] = v1[0]*v2[1] - v1[1]*v2[0]
    return res

def main():
    robot = MBot()
    robot.reset_odometry()
    time.sleep(0.5)

    try:
        goal_x = float(input("Goal X: "))
        goal_y = float(input("Goal Y: "))
        goal_theta = float(input("Goal Theta: "))
    except ValueError:
        print("Invalid input. Defaulting to (1, 0, 0)")
        goal_x, goal_y, goal_theta = 1.0, 0.0, 0.0

    STATE_GO_TO_GOAL = 0
    STATE_WALL_FOLLOW = 1
    
    current_state = STATE_GO_TO_GOAL
    
    OBSTACLE_DIST = 0.5   
    CLEAR_DIST = 0.7      
    SLICE_SIZE = np.deg2rad(60) 
    GOAL_TOLERANCE = 0.05

    print("Starting Bug Navigation...")

    try:
        while True:
            ranges, thetas = robot.read_lidar()
            pose = robot.read_odometry()
            x, y, theta = pose[0], pose[1], pose[2]

            dx = goal_x - x
            dy = goal_y - y
            dist_to_goal = np.sqrt(dx**2 + dy**2)
            
            global_angle = np.arctan2(dy, dx)
            robot_angle_to_goal = normalize_angle(global_angle - theta)
            
            slice_min_dist = find_min_ray_in_slice(ranges, thetas, robot_angle_to_goal, SLICE_SIZE)

            print(f"State: {current_state} | Goal Dist: {dist_to_goal:.2f} | Obs Dist: {slice_min_dist:.2f}")

            if current_state == STATE_GO_TO_GOAL:
                if dist_to_goal < GOAL_TOLERANCE:
                    print("Goal Reached!")
                    break
                if slice_min_dist < OBSTACLE_DIST:
                    print("--> Blocked! Switching to Wall Follow")
                    current_state = STATE_WALL_FOLLOW
            
            elif current_state == STATE_WALL_FOLLOW:
                if slice_min_dist > CLEAR_DIST:
                    print("--> Clear! Switching to Go To Goal")
                    current_state = STATE_GO_TO_GOAL

            if current_state == STATE_GO_TO_GOAL:
                vx = 1.0 * dist_to_goal
                wz = 4.0 * robot_angle_to_goal
                if abs(robot_angle_to_goal) > np.pi/2: vx = 0
                
                vx = np.clip(vx, -0.4, 0.4)
                wz = np.clip(wz, -1.5, 1.5)
                robot.drive(vx, 0, wz)

            elif current_state == STATE_WALL_FOLLOW:
                min_dist, min_angle = find_min_dist_all(ranges, thetas)
                
                if min_dist > 2.0 or min_dist == 0:
                    robot.drive(0.05, 0, 0.5)
                else:
                    setpoint = 0.3
                    k_p = 3.0
                    
                    wall_vec = np.array([np.cos(min_angle), np.sin(min_angle), 0.0])
                    z_axis = np.array([0, 0, 1])
                    forward_vec = cross_product(z_axis, wall_vec)
                    forward_vec = forward_vec / np.linalg.norm(forward_vec[:2])
                    
                    error = min_dist - setpoint
                    correction = k_p * error
                    
                    robot.drive(0.25, 0, -correction)

            time.sleep(0.05)

    except KeyboardInterrupt:
        robot.stop()
    
    robot.stop()
    print(f"Final Pose: {robot.read_odometry()}")

if __name__ == "__main__":
    main()