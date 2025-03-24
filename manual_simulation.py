import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import time
from surgical_robot_arm import SurgicalRobotArm

robot = SurgicalRobotArm()

# Create a figure and 3D axis
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Function to update the plot
def update_plot(angles):
    ax.clear()
    positions = robot.forward_kinematics(angles)
    
    # Extract x, y, z coordinates for plotting
    x_coords = [p[0] for p in positions]
    y_coords = [p[1] for p in positions]
    z_coords = [p[2] for p in positions]
    
    # Plot the robot arm links
    ax.plot(x_coords, y_coords, z_coords, 'bo-', linewidth=2, markersize=6)
    
    # Plot the end effector
    ax.scatter(x_coords[-1], y_coords[-1], z_coords[-1], color='red', s=100)
    
    # Set axis labels and limits
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    max_range = 0.3
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.set_zlim([0, max_range])
    
    ax.set_title('4-DOF Robotic Arm - Manual Control')

# Function to handle keyboard input
def on_key(event):
    # Adjust joint angles based on key press
    if event.key == 'up':
        robot.current_angles[1] += 0.1  # Increase Joint 2 angle
    elif event.key == 'down':
        robot.current_angles[1] -= 0.1  # Decrease Joint 2 angle
    elif event.key == 'left':
        robot.current_angles[0] += 0.1  # Increase Joint 1 angle
    elif event.key == 'right':
        robot.current_angles[0] -= 0.1  # Decrease Joint 1 angle
    elif event.key == 'w':
        robot.current_angles[2] += 0.1  # Increase Joint 3 angle
    elif event.key == 's':
        robot.current_angles[2] -= 0.1  # Decrease Joint 3 angle
    elif event.key == 'a':
        robot.current_angles[3] += 0.1  # Increase Joint 4 angle
    elif event.key == 'd':
        robot.current_angles[3] -= 0.1  # Decrease Joint 4 angle
    
    # Update the plot with new angles
    update_plot(robot.current_angles)
    plt.draw()

# Connect the keyboard event handler
fig.canvas.mpl_connect('key_press_event', on_key)

# Initial plot
update_plot(robot.current_angles)
plt.show()