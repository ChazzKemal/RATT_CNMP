# Description: This file is used to control the end effector position using keyboard keys.
import keyboard
import math
from scipy.ndimage import zoom

from environment_termproject import *


def replay_demonstration(demonstration, env): # also see env._follow_ee_trajectory(position_traj, orientation_traj)
    states = []
    rewards = []
    for ee_State in demonstration:
        env._set_ee_pose(np.array((ee_State[0], ee_State[1], 1.07)), rotation=[-90, 0, 180], max_iters=10)
        
        state = env.high_level_state()
        states.append(state)
        
        reward = env.reward()
        rewards.append(reward)
        
        ee_State = state[:3]
        print(f"End effector position: {ee_State}")
    
    states = np.array(states)
    rewards = np.array(rewards)
    
    new_demonstration = np.hstack((states, rewards.reshape(-1, 1)))
    
    return new_demonstration


def interpolate_demonstration(demonstration, steps=100):
    interpolated_demonstration = []
    t = np.linspace(0, 1, steps).reshape(-1, 1)
    demonstration_x = demonstration[:, 0]
    demonstration_y = demonstration[:, 1]
    
    for arr in [demonstration_x, demonstration_y]:
        zoom_rate = t.shape[0] / arr.shape[0]
        arr = zoom(arr, zoom_rate)
        #print(arr.shape)
        interpolated_demonstration.append(arr)
    
    interpolated_demonstration = np.column_stack((interpolated_demonstration[0], interpolated_demonstration[1]))
    
    return interpolated_demonstration
    
    
#GLOBAL VARS FOR ENVIRONMENT:
flag_rand = False
obj_position = np.array((0.70, -0.10))
goal_position = np.array((0.30, 0.15))
origin = np.array((0.5, 0.3, 1.07))
num_of_demos = 20

env = TermProject_env(obj_position=obj_position, goal_position=goal_position, flag_rand=flag_rand, render_mode="gui")

if __name__ == "__main__":    

    step_size = 0.005
    flag_print = False
    
    for i in range(num_of_demos):
        current_states= []
        rewards = []
        env.reset()
        
        # Setting initial position; i.e., origin point
        env._set_ee_in_cartesian(origin, rotation=[-90, 0, 180], n_splits=100, max_iters=100, threshold=0.05)

        #Getting initial states
        state = env.high_level_state()
        ee_State = state[:3]
        current_states.append(state)

        rewards.append(env.reward())
        print(f"İnitial effector position: {ee_State}")
        
        
        while True:
            try:
                if keyboard.is_pressed('esc'):
                    break
                elif keyboard.is_pressed('6'): # not properly working?
                    flag_print = True
                    ee_State[0] -= step_size
                    ee_State[1] += step_size
                elif keyboard.is_pressed('4'): # not properly working?
                    flag_print = True
                    ee_State[0] -= step_size
                    ee_State[1] -= step_size
                elif keyboard.is_pressed('9'):
                    flag_print = True
                    ee_State[1] += step_size
                elif keyboard.is_pressed('7'):
                    flag_print = True
                    ee_State[1] -= step_size
                elif keyboard.is_pressed('8'):
                    flag_print = True
                    ee_State[0] -= step_size
                elif keyboard.is_pressed('2'):
                    flag_print = True
                    ee_State[0] += step_size
                elif keyboard.is_pressed('1'):
                    flag_print = True
                    ee_State[0] += step_size
                    ee_State[1] -= step_size
                elif keyboard.is_pressed('3'):
                    flag_print = True
                    ee_State[0] += step_size
                    ee_State[1] += step_size
                    
                if flag_print:
                        env._set_ee_pose(np.array((ee_State[0], ee_State[1], 1.07)), rotation=[-90, 0, 180], max_iters=10)
                        state = env.high_level_state()
                        ee_State = state[:3]
                        
                        # Saving states
                        current_states.append(state)
                        rewards.append(env.reward())
                        
                        # Printing state EE
                        print(f"End effector position: {ee_State}")
                        flag_print = False
                        
            except KeyboardInterrupt:
                print("Keyboard interrupt!")
                break
        
        current_states = np.array(current_states)
        rewards = np.array(rewards)
        
        current_demonstration = np.hstack((current_states, rewards.reshape(-1, 1)))
        demonstrations.append(current_demonstration)
        # np.save(f"demo_{num_of_demos}.npy", demonstrations[0])
        # print(f"Demo {num_of_demos} is saved!")
        np.save(f"demos/demo_{i+1}.npy", demonstrations[i])
        print(f"Demo {i+1} is saved!")