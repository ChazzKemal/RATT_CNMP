# Description: This file is used to control the end effector position using keyboard keys.
import keyboard

from environment_termproject import *

#GLOBAL VARS FOR ENVIRONMENT:
flag_rand = False
obj_position = np.array((0.65, -0.10))
goal_position = np.array((0.40, 0.15))
origin = np.array((0.6, 0.3, 1.07))
num_of_demos = 40
object_dentsity = 5000

env = TermProject_env(origin=origin, obj_position=obj_position, goal_position=goal_position, flag_rand=flag_rand, density=object_dentsity, render_mode="gui")

if __name__ == "__main__":    
    step_size = 0.00751
    record_flag = False
    
    for i in range(num_of_demos):
        current_states= []
        rewards = []
        env.reset()
        
        # Go go origin and close gripper
        env.go_origin()
        env.close_gripper()
        
        #Getting initial states
        state = env.high_level_state()
        ee_State = state[:2]
        current_states.append(state)

        rewards.append(env.reward())
        #print(f"İnitial effector position: {ee_State}")
        
        while True:
            try:
                if keyboard.is_pressed('esc'):
                    break
                elif keyboard.is_pressed('6'): # not properly working?
                    record_flag = True
                    ee_State[0] -= step_size
                    ee_State[1] += step_size
                elif keyboard.is_pressed('4'): # not properly working?
                    record_flag = True
                    ee_State[0] -= step_size
                    ee_State[1] -= step_size
                elif keyboard.is_pressed('9'):
                    record_flag = True
                    ee_State[1] += step_size
                elif keyboard.is_pressed('7'):
                    record_flag = True
                    ee_State[1] -= step_size
                elif keyboard.is_pressed('8'):
                    record_flag = True
                    ee_State[0] -= step_size
                elif keyboard.is_pressed('2'):
                    record_flag = True
                    ee_State[0] += step_size
                elif keyboard.is_pressed('1'):
                    record_flag = True
                    ee_State[0] += step_size
                    ee_State[1] -= step_size
                elif keyboard.is_pressed('3'):
                    record_flag = True
                    ee_State[0] += step_size
                    ee_State[1] += step_size
                    
                if record_flag:
                        env._set_ee_pose(np.array((ee_State[0], ee_State[1], 1.07)), rotation=[-90, 0, 180], max_iters=10)
                        state = env.high_level_state()
                        ee_State = state[:2]
                        
                        # Saving states
                        current_states.append(state)
                        rewards.append(env.reward())
                        
                        # Printing state EE
                        #print(f"End effector position: {ee_State}")
                        record_flag = False
                        
            except KeyboardInterrupt:
                print("Keyboard interrupt!")
                break
        
        current_states = np.array(current_states)
    
        np.save(f"demos/demo_{i+1}.npy", current_states)
        print(f"Demo {i+1} is saved!")