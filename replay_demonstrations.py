from env_keyop_demonstration import *

from matplotlib import pyplot as plt

if __name__ == "__main__":
    for i in range(num_of_demos):
        env.reset()
        
        loaded_demo = np.load(f"demos/demo_{i+1}.npy")
        ee_States = loaded_demo[:, :3]
        
        #####################################################################
        #Reward correction for initial commit, deprecated
        # rewards =[]
        # for state in loaded_demo:
        #     ee_pos = state[:2] # 3rd index is height
        #     obj_pos = state[3:5]
        #     goal_pos = state[5:7]
        #     ee_to_obj = max(100*np.linalg.norm(ee_pos - obj_pos), 1)
        #     obj_to_goal = max(100*np.linalg.norm(obj_pos - goal_pos), 1)
        #     current_  = 1/(ee_to_obj) + 1/(obj_to_goal)
        #     rewards.append(current_reward)
        
        # loaded_demo[:, 7] = rewards
        # np.save(f"demos/demo_{i+1}.npy", loaded_demo)
        #####################################################################
        
        curve=interpolate_demonstration(ee_States, steps=100)
        interpolated_demonstrations=replay_demonstration(curve, env)
        
        #Saving interpolated demonstrations -> timestep=100
        np.save(f"demos/demo_{i+1}.npy", interpolated_demonstrations)
        
        # Plot the demonstrations
        #plt.plot(ee_States[:, 0], ee_States[:, 1], label=f"demo_{i+1}")