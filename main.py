import time
from surgical_robot_arm import SurgicalRobotArm

def main():
    """Main function to run the 4-DOF Surgical Robot Arm simulation."""
    print("Initializing 4-DOF Surgical Robot Arm Simulation...")
    robot = SurgicalRobotArm()
    
    # Run all demonstrations
    robot.run_demonstrations()
    
    print("\nSimulation complete!")

if __name__ == "__main__":
    main()