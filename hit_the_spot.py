import time
import numpy as np
from mbot_bridge.api import MBot



def drive_to_pose(goal_x, goal_y, goal_theta, robot):
  def normalize_angle(angle):
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle < -np.pi:
        angle += 2 * np.pi
    return angle

  K_p_lin = 1.5  
  K_p_ang = 4.0  
  
  DIST_THRESHOLD = 0.05 
  ANGLE_THRESHOLD = 0.05 
  
  print(f"Navigating to: ({goal_x}, {goal_y}, {goal_theta})")

  while True:
      pose = robot.read_odometry()
      x, y, theta = pose[0], pose[1], pose[2]

      dx = goal_x - x
      dy = goal_y - y
      distance_error = np.sqrt(dx**2 + dy**2)
      target_heading = np.arctan2(dy, dx)
      heading_error = normalize_angle(target_heading - theta)

      if distance_error > DIST_THRESHOLD:
          vx = K_p_lin * distance_error
          wz = K_p_ang * heading_error
          
          if abs(heading_error) > np.pi / 2:
              vx = 0.0
      else:
          final_angle_error = normalize_angle(goal_theta - theta)
          vx = 0.0
          wz = K_p_ang * final_angle_error
          
          if abs(final_angle_error) < ANGLE_THRESHOLD:
              print("Target Reached.")
              robot.stop()
              break

      vx = np.clip(vx, -0.5, 0.5)
      wz = np.clip(wz, -2.0, 2.0)

      robot.drive(vx, 0, wz)
      time.sleep(0.05)


if __name__ == "__main__":
    robot = MBot()
    robot.reset_odometry()
    time.sleep(1)

    try:
        drive_to_pose(1.0, 0.0, 0.0, robot)
        time.sleep(1)
        drive_to_pose(0.0, 0.0, 0.0, robot)
        
    except KeyboardInterrupt:
        robot.stop()