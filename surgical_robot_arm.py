import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import time

class SurgicalRobotArm:
    def __init__(self):
        # Define DH parameters [alpha, a, d, theta]
        # For a 4-DOF arm designed for minimally invasive surgery
        self.dh_params = [
            [0, 0, 0.1, 0],           # Base to joint 1 (revolute)
            [np.pi/2, 0, 0, 0],       # Joint 1 to joint 2 (revolute)
            [0, 0.12, 0, 0],          # Joint 2 to joint 3 (revolute)
            [0, 0.1, 0, 0]            # Joint 3 to end effector (revolute)
        ]
        
        # Joint limits (min, max) in radians
        self.joint_limits = [
            [-np.pi, np.pi],      # Joint 1 (base rotation)
            [-np.pi/2, np.pi/2],  # Joint 2
            [-3*np.pi/4, 3*np.pi/4], # Joint 3
            [-np.pi, np.pi]       # Joint 4 (end effector)
        ]
        
        # Current joint angles
        self.current_angles = [0, 0, 0, 0]
        
    def transformation_matrix(self, alpha, a, d, theta):
        """Calculate the transformation matrix based on DH parameters."""
        # Convert angles to radians if needed
        alpha_rad = alpha
        theta_rad = theta
        
        # Create the transformation matrix
        T = np.array([
            [np.cos(theta_rad), -np.sin(theta_rad)*np.cos(alpha_rad), np.sin(theta_rad)*np.sin(alpha_rad), a*np.cos(theta_rad)],
            [np.sin(theta_rad), np.cos(theta_rad)*np.cos(alpha_rad), -np.cos(theta_rad)*np.sin(alpha_rad), a*np.sin(theta_rad)],
            [0, np.sin(alpha_rad), np.cos(alpha_rad), d],
            [0, 0, 0, 1]
        ])
        
        return T
    
    def forward_kinematics(self, joint_angles=None):
        """Calculate the forward kinematics for the robot arm."""
        if joint_angles is None:
            joint_angles = self.current_angles
            
        # Ensure joint angles are within limits
        for i in range(len(joint_angles)):
            joint_angles[i] = max(self.joint_limits[i][0], min(joint_angles[i], self.joint_limits[i][1]))
        
        # Start with the identity matrix
        T = np.identity(4)
        
        # Calculate positions of each joint
        positions = [np.array([0, 0, 0])]  # Start with base position
        
        # Make a deep copy of DH parameters to avoid modifying the original
        dh_copy = [param.copy() for param in self.dh_params]
        
        # Update DH parameters with current joint angles
        for i in range(len(joint_angles)):
            dh_copy[i][3] = joint_angles[i]
            
            # Calculate transformation matrix for this joint
            Ti = self.transformation_matrix(dh_copy[i][0], dh_copy[i][1], dh_copy[i][2], dh_copy[i][3])
            
            # Update the cumulative transformation
            T = T @ Ti
            
            # Extract the position from the transformation matrix
            position = T[:3, 3]
            positions.append(position)
        
        # Update current angles if this was a direct call
        if joint_angles is not self.current_angles:
            self.current_angles = joint_angles.copy()
        
        return positions
    
    def inverse_kinematics(self, target_position):
        """Calculate joint angles to reach the target position."""
        x, y, z = target_position
        
        # Extract link parameters
        d1 = self.dh_params[0][2]  # Height of first link
        a2 = self.dh_params[2][1]  # Length of second link
        a3 = self.dh_params[3][1]  # Length of third link
        
        # Calculate joint 1 (base rotation)
        theta1 = np.arctan2(y, x)
        
        # Distance from base to target in xy plane
        r = np.sqrt(x**2 + y**2)
        
        # Calculate the distance to the target in the plane of joints 2 and 3
        z_prime = z - d1
        
        # Distance from joint 2 to target
        D = np.sqrt(r**2 + z_prime**2)
        
        # Check if target is reachable
        if D > a2 + a3:
            print(f"Warning: Target position {target_position} is out of reach.")
            # Return the current angles unchanged
            return self.current_angles
        
        # Use law of cosines to find angles
        # Angle for joint 3
        cos_theta3 = (D**2 - a2**2 - a3**2) / (2 * a2 * a3)
        # Constrain within valid range
        cos_theta3 = np.clip(cos_theta3, -1.0, 1.0)
        theta3 = np.arccos(cos_theta3)
        
        # Angle for joint 2
        beta = np.arctan2(z_prime, r)
        gamma = np.arctan2(a3 * np.sin(theta3), a2 + a3 * np.cos(theta3))
        theta2 = beta - gamma
        
        # Joint 4 (end effector orientation)
        # For surgical applications, we want the end effector to be perpendicular to the tissue surface
        # This is a simplified approach; in reality, this would depend on the surgical task
        theta4 = -(theta2 + theta3)  # This creates a horizontal end effector
        
        # Apply joint limits
        theta1 = max(self.joint_limits[0][0], min(theta1, self.joint_limits[0][1]))
        theta2 = max(self.joint_limits[1][0], min(theta2, self.joint_limits[1][1]))
        theta3 = max(self.joint_limits[2][0], min(theta3, self.joint_limits[2][1]))
        theta4 = max(self.joint_limits[3][0], min(theta4, self.joint_limits[3][1]))
        
        return [theta1, theta2, theta3, theta4]
    
    def move_to_target(self, target_position):
        """Move the arm to a target position using inverse kinematics."""
        # Calculate the joint angles needed to reach the target
        target_angles = self.inverse_kinematics(target_position)
        
        # Update the current angles
        self.current_angles = target_angles
        
        # Verify reaching the target by calculating forward kinematics
        positions = self.forward_kinematics()
        
        # Return the end effector position
        return positions[-1]
    
    def visualize_robot(self, positions=None):
        """Visualize the robot arm in 3D space."""
        if positions is None:
            positions = self.forward_kinematics()
            
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Extract x, y, z coordinates for plotting
        x_coords = [p[0] for p in positions]
        y_coords = [p[1] for p in positions]
        z_coords = [p[2] for p in positions]
        
        # Plot the robot arm links
        ax.plot(x_coords, y_coords, z_coords, 'bo-', linewidth=2, markersize=6)
        
        # Plot the end effector
        ax.scatter(x_coords[-1], y_coords[-1], z_coords[-1], color='red', s=100, marker='o')
        
        # Set axis labels
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        
        # Set axis limits with some margin
        max_range = np.max([
            np.max(np.abs(x_coords)),
            np.max(np.abs(y_coords)),
            np.max(np.abs(z_coords))
        ]) * 1.2
        
        if max_range < 0.3:
            max_range = 0.3
        
        ax.set_xlim([-max_range, max_range])
        ax.set_ylim([-max_range, max_range])
        ax.set_zlim([0, max_range])
        
        ax.set_title('4-DOF Robotic Arm for Minimally Invasive Surgery')
        
        plt.tight_layout()
        return fig, ax
    
    def demonstrate_forward_kinematics(self):
        """Demonstrate the forward kinematics of the robot arm."""
        # Create figures for multiple poses
        plt.figure(figsize=(15, 10))
        
        # Show 4 different arm configurations
        test_angles = [
            [0, 0, 0, 0],
            [np.pi/4, np.pi/4, 0, 0],
            [np.pi/2, np.pi/4, np.pi/4, 0],
            [np.pi/4, np.pi/3, np.pi/6, np.pi/4]
        ]
        
        for i, angles in enumerate(test_angles):
            positions = self.forward_kinematics(angles)
            
            ax = plt.subplot(2, 2, i+1, projection='3d')
            
            # Extract x, y, z coordinates for plotting
            x_coords = [p[0] for p in positions]
            y_coords = [p[1] for p in positions]
            z_coords = [p[2] for p in positions]
            
            # Plot the robot arm links
            ax.plot(x_coords, y_coords, z_coords, 'bo-', linewidth=2, markersize=6)
            
            # Plot the end effector
            ax.scatter(x_coords[-1], y_coords[-1], z_coords[-1], color='red', s=100)  # Corrected line
            
            ax.set_title(f'Configuration {i+1}: θ={[round(a*180/np.pi) for a in angles]} degrees')
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_zlabel('Z (m)')
            
            max_range = 0.3
            ax.set_xlim([-max_range, max_range])
            ax.set_ylim([-max_range, max_range])
            ax.set_zlim([0, max_range])
        
        plt.tight_layout()
        plt.savefig('forward_kinematics_demo.png')
        plt.show()
        
        return 'Forward kinematics demonstration completed and saved as forward_kinematics_demo.png'


    def demonstrate_inverse_kinematics(self):
        """Demonstrate the inverse kinematics of the robot arm."""
        # Define target positions
        target_positions = [
            [0.1, 0.1, 0.15],  # Target 1
            [0.15, 0, 0.12],   # Target 2
            [0.05, 0.15, 0.1], # Target 3
            [0.12, 0.12, 0.18] # Target 4
        ]
        
        plt.figure(figsize=(15, 10))
        
        for i, target in enumerate(target_positions):
            # Calculate joint angles using inverse kinematics
            joint_angles = self.inverse_kinematics(target)
            
            # Calculate forward kinematics to verify
            positions = self.forward_kinematics(joint_angles)
            
            # Plot
            ax = plt.subplot(2, 2, i+1, projection='3d')
            
            # Extract coordinates
            x_coords = [p[0] for p in positions]
            y_coords = [p[1] for p in positions]
            z_coords = [p[2] for p in positions]
            
            # Plot the robot arm
            ax.plot(x_coords, y_coords, z_coords, 'bo-', linewidth=2, markersize=6)
            
            # Plot the end effector
            ax.scatter(x_coords[-1], y_coords[-1], z_coords[-1], color='red', s=100)  # Corrected line
            
            # Plot the target position
            ax.scatter(target[0], target[1], target[2], color='green', s=100, marker='x')
            
            ax.set_title(f'Target Position {i+1}: ({target[0]:.2f}, {target[1]:.2f}, {target[2]:.2f})')
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_zlabel('Z (m)')
            
            max_range = 0.3
            ax.set_xlim([-max_range, max_range])
            ax.set_ylim([-max_range, max_range])
            ax.set_zlim([0, max_range])
        
        plt.tight_layout()
        plt.savefig('inverse_kinematics_demo.png')
        plt.show()
        
        return 'Inverse kinematics demonstration completed and saved as inverse_kinematics_demo.png'
        
    def animate_motion(self, start_angles, end_angles, title, save_gif=True):
        """Animate the motion of the robot arm from start to end joint angles."""
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Number of frames for the animation
        frames = 50
        
        def update(frame):
            ax.clear()
            
            # Interpolate joint angles
            t = frame / frames
            current_angles = start_angles + t * (np.array(end_angles) - np.array(start_angles))
            
            # Calculate positions
            positions = self.forward_kinematics(current_angles)
            
            # Extract coordinates
            x_coords = [p[0] for p in positions]
            y_coords = [p[1] for p in positions]
            z_coords = [p[2] for p in positions]
            
            # Plot the robot arm
            ax.plot(x_coords, y_coords, z_coords, 'bo-', linewidth=2, markersize=6)
            
            # Plot the end effector
            ax.scatter(x_coords[-1], y_coords[-1], z_coords[-1], color='red', s=100, marker='o')
            
            ax.set_title(title)
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_zlabel('Z (m)')
            
            max_range = 0.3
            ax.set_xlim([-max_range, max_range])
            ax.set_ylim([-max_range, max_range])
            ax.set_zlim([0, max_range])
        
        ani = FuncAnimation(fig, update, frames=frames, interval=50)
        plt.tight_layout()
        
        # Save animation as a GIF
        if save_gif:
            filename = f'{title.replace(" ", "_")}.gif'
            ani.save(filename, writer='pillow')
            print(f"Animation saved as {filename}")
        
        plt.show()
    
    def demonstrate_surgical_path(self):
        """Demonstrate the robot following a surgical path."""
        # Define a surgical path (e.g., a suturing pattern)
        surgical_path = [
            [0.15, 0, 0.15],    # Initial position
            [0.15, 0.05, 0.12], # Insert
            [0.17, 0.05, 0.12], # Move right
            [0.17, 0, 0.12],    # Move back
            [0.15, 0, 0.15]     # Return to initial position
        ]
        
        plt.figure(figsize=(12, 8))
        ax = plt.subplot(111, projection='3d')
        
        # Plot the surgical path
        path_x = [p[0] for p in surgical_path]
        path_y = [p[1] for p in surgical_path]
        path_z = [p[2] for p in surgical_path]
        
        ax.plot(path_x, path_y, path_z, 'g--', linewidth=2, label='Surgical Path')
        
        # Calculate joint angles for each point in the path
        for i, target in enumerate(surgical_path):
            joint_angles = self.inverse_kinematics(target)
            positions = self.forward_kinematics(joint_angles)
            
            # Extract coordinates
            x_coords = [p[0] for p in positions]
            y_coords = [p[1] for p in positions]
            z_coords = [p[2] for p in positions]
            
            # Plot the robot arm at this position with transparency
            alpha = 0.3 if i != 0 and i != len(surgical_path) - 1 else 0.8
            ax.plot(x_coords, y_coords, z_coords, 'bo-', linewidth=2, markersize=6, alpha=alpha)
            
            # Plot the end effector
            ax.scatter(x_coords[-1], y_coords[-1], z_coords[-1], color='red', s=100, marker='o', alpha=alpha)
            
        # Annotate the path points
        labels = ['Initial', 'Insert', 'Move Right', 'Move Back', 'Return']
        for i, (x, y, z) in enumerate(surgical_path):
            ax.text(x, y, z, f'{labels[i]}', fontsize=10)
        
        ax.set_title('Surgical Path Demonstration')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        
        max_range = 0.3
        ax.set_xlim([-max_range, max_range])
        ax.set_ylim([-max_range, max_range])
        ax.set_zlim([0, max_range])
        
        ax.legend()
        plt.tight_layout()
        plt.savefig('surgical_path_demo.png')
        plt.show()
        
        return 'Surgical path demonstration completed and saved as surgical_path_demo.png'

    def animate_surgical_path(self):
        """Animate the robot following a surgical path."""
        # Define a surgical path
        surgical_path = [
            [0.15, 0, 0.15],    # Initial position
            [0.15, 0.05, 0.12], # Insert
            [0.17, 0.05, 0.12], # Move right
            [0.17, 0, 0.12],    # Move back
            [0.15, 0, 0.15]     # Return to initial position
        ]
        
        # Calculate joint angles for each position
        joint_angle_sequence = []
        for target in surgical_path:
            joint_angles = self.inverse_kinematics(target)
            joint_angle_sequence.append(joint_angles)
        
        # Create animation
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot the surgical path
        path_x = [p[0] for p in surgical_path]
        path_y = [p[1] for p in surgical_path]
        path_z = [p[2] for p in surgical_path]
        
        # Number of frames between each waypoint
        frames_between = 20
        total_points = len(surgical_path)
        
        # Interpolate between waypoints
        def interpolate(p1, p2, t):
            return p1 + t * (p2 - p1)
        
        def update(frame):
            ax.clear()
            
            # Plot the path
            ax.plot(path_x, path_y, path_z, 'g--', linewidth=2, label='Surgical Path')
            
            # Determine which segment we're in
            segment = min(int(frame / frames_between), total_points - 2)
            t = (frame % frames_between) / frames_between
            
            # Interpolate joint angles
            start_angles = np.array(joint_angle_sequence[segment])
            end_angles = np.array(joint_angle_sequence[segment + 1])
            current_angles = interpolate(start_angles, end_angles, t)
            
            # Calculate positions
            positions = self.forward_kinematics(current_angles)
            
            # Extract coordinates
            x_coords = [p[0] for p in positions]
            y_coords = [p[1] for p in positions]
            z_coords = [p[2] for p in positions]
            
            # Plot the robot arm
            ax.plot(x_coords, y_coords, z_coords, 'bo-', linewidth=2, markersize=6)
            
            # Plot the end effector
            ax.scatter(x_coords[-1], y_coords[-1], z_coords[-1], color='red', s=100, marker='o')
            
            # Annotate the path points
            labels = ['Initial', 'Insert', 'Move Right', 'Move Back', 'Return']
            for i, (x, y, z) in enumerate(surgical_path):
                ax.text(x, y, z, f'{labels[i]}', fontsize=8)
            
            ax.set_title('Surgical Path Animation')
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_zlabel('Z (m)')
            
            max_range = 0.3
            ax.set_xlim([-max_range, max_range])
            ax.set_ylim([-max_range, max_range])
            ax.set_zlim([0, max_range])
            
            ax.legend()
        
        frames = (total_points - 1) * frames_between
        ani = FuncAnimation(fig, update, frames=frames, interval=50)
        plt.tight_layout()
        
        # Save animation
        ani.save('surgical_path_animation.gif', writer='pillow')
        plt.show()
        
        return 'Surgical path animation completed and saved as surgical_path_animation.gif'

    def workspace_analysis(self):
        """Analyze the workspace of the robot."""
        # Create a grid of points to check
        resolution = 10
        x_range = np.linspace(-0.3, 0.3, resolution)
        y_range = np.linspace(-0.3, 0.3, resolution)
        z_range = np.linspace(0, 0.3, resolution)
        
        # Arrays to store reachable points
        reachable_points = []
        
        # Check each point in the grid
        for x in x_range:
            for y in y_range:
                for z in z_range:
                    target = [x, y, z]
                    
                    # Try to reach the point
                    joint_angles = self.inverse_kinematics(target)
                    positions = self.forward_kinematics(joint_angles)
                    
                    # Check if we actually reached the target
                    end_effector = positions[-1]
                    distance = np.sqrt((end_effector[0] - x)**2 + 
                                    (end_effector[1] - y)**2 + 
                                    (end_effector[2] - z)**2)
                    
                    # If we're close enough, consider it reachable
                    if distance < 0.01:
                        reachable_points.append(target)
        
        # Plot the workspace
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Extract coordinates of reachable points
        if reachable_points:
            rx = [p[0] for p in reachable_points]
            ry = [p[1] for p in reachable_points]
            rz = [p[2] for p in reachable_points]
            
            # Plot the reachable points
            ax.scatter(rx, ry, rz, c='blue', alpha=0.5, label='Reachable Points')
        
        ax.set_title('Robot Workspace Analysis')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        
        max_range = 0.3
        ax.set_xlim([-max_range, max_range])
        ax.set_ylim([-max_range, max_range])
        ax.set_zlim([0, max_range])
        
        ax.legend()
        plt.tight_layout()
        plt.savefig('workspace_analysis.png')
        plt.show()
        
        return f'Workspace analysis completed. Found {len(reachable_points)} reachable points.'

    def demo_homogeneous_transformation_matrices(self):
        """Demonstrate the calculation of homogeneous transformation matrices."""
        # Use current angles
        joint_angles = self.current_angles
        
        # Make a deep copy of DH parameters
        dh_copy = [param.copy() for param in self.dh_params]
        
        # Update DH parameters with current joint angles
        for i in range(len(joint_angles)):
            dh_copy[i][3] = joint_angles[i]
        
        print("DH Parameters with current joint angles:")
        print("    α       a       d       θ")
        for i, param in enumerate(dh_copy):
            print(f"J{i+1}: {param[0]:.4f}  {param[1]:.4f}  {param[2]:.4f}  {param[3]:.4f}")
        
        print("\nIndividual Transformation Matrices:")
        T_cumulative = np.identity(4)
        
        for i in range(len(dh_copy)):
            Ti = self.transformation_matrix(dh_copy[i][0], dh_copy[i][1], dh_copy[i][2], dh_copy[i][3])
            print(f"\nT{i}_{i+1}:")
            print(np.array2string(Ti, precision=4, suppress_small=True))
            
            T_cumulative = T_cumulative @ Ti
            
            print(f"\nT0_{i+1} (Cumulative):")
            print(np.array2string(T_cumulative, precision=4, suppress_small=True))
        
        print("\nFinal Homogeneous Transformation Matrix:")
        print(np.array2string(T_cumulative, precision=4, suppress_small=True))
        
        # Extract position and orientation
        position = T_cumulative[:3, 3]
        rotation = T_cumulative[:3, :3]
        
        print("\nEnd Effector Position (x, y, z):")
        print(f"({position[0]:.4f}, {position[1]:.4f}, {position[2]:.4f})")
        
        return "Homogeneous transformation matrices demonstrated."

    def run_demonstrations(self):
        """Run all the demonstrations."""
        # Demonstrate forward kinematics
        print("\n=== Demonstrating Forward Kinematics ===")
        self.demonstrate_forward_kinematics()
        
        # Demonstrate inverse kinematics
        print("\n=== Demonstrating Inverse Kinematics ===")
        self.demonstrate_inverse_kinematics()
        
        # Demonstrate a surgical path
        print("\n=== Demonstrating Surgical Path ===")
        self.demonstrate_surgical_path()
        
        # Animate a surgical path
        print("\n=== Animating Surgical Path ===")
        self.animate_surgical_path()
        
        # Show homogeneous transformation matrices
        print("\n=== Demonstrating Homogeneous Transformation Matrices ===")
        self.demo_homogeneous_transformation_matrices()
        
        print("\nAll demonstrations completed!")