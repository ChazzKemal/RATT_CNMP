import numpy as np

from environment_termproject import TermProject_env
from replay_demonstrations import interpolate_demonstration, replay_demonstration

#GLOBAL VARS FOR ENVIRONMENT:
flag_rand = False
obj_position = np.array((0.65, -0.10))
goal_position = np.array((0.40, 0.15))
origin = np.array((0.6, 0.3, 1.07))
num_of_demos = 40
object_dentsity = 5000

# Creating environment with given parameters

def main():
    timestep = 300
    env = TermProject_env(origin= origin, obj_position=obj_position, goal_position=goal_position, flag_rand=flag_rand, density=object_dentsity, render_mode="gui")
    
    env.reset()
    
    # Movin the robot to initial position
    env.go_origin()
    
    # Closing the gripper
    env.close_gripper()
    
    # Loading demonstrations
    demo = np.load("demos/demo_1.npy")
    demo_int = interpolate_demonstration(demo, steps=timestep)
    
    # Moving the robot according to the demonstration
    new_demo = replay_demonstration(demo_int, demo, env)
    
    
if __name__ == "__main__":
    main()