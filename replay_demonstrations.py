from record_demonstration import *

from scipy.ndimage import zoom


def replay_demonstration(demonstration_int, demonstration, env): # also see env._follow_ee_trajectory(position_traj, orientation_traj)
    states = []
    rewards = []
    time_series = []
    for i, ee_State in enumerate(demonstration_int):
        #time_series
        t = i/len(demonstration_int)
        time_series.append(t)
        
        #record states
        state = env.high_level_state()
        
        ee_curr_State = np.array(state[:2]) # X, Y
        obj_curr_State = np.array(state[2:4]) # X, Y
        goal_curr_State = np.array(state[4:]) # X, Y
        
        # print(f"ee_curr_State: {ee_curr_State}")
        # print(f"obj_curr_State: {obj_curr_State}")
        # print(f"goal_curr_State: {goal_curr_State}")
        
        new_State = np.hstack((goal_curr_State, obj_curr_State, ee_curr_State))
        states.append(new_State)
        
        #reward
        reward = term_project_reward(demonstration, state)
        rewards.append(reward)
        
        # Move the robot to the next position
        env._set_ee_pose(np.array((ee_State[0], ee_State[1], 1.07)), rotation=[-90, 0, 180], max_iters=10)
        
    states = np.array(states).reshape(-1, 6)
    rewards = np.array(rewards).reshape(-1, 1)
    time_series = np.array(time_series).reshape(-1, 1)
    
    # print(f"states: {states.shape}")
    # print(f"rewards: {rewards.shape}")
    # print(f"time_series: {time_series.shape}")
    
    new_demonstration = np.concatenate((time_series, rewards, states), axis=1)
    
    return new_demonstration

def term_project_reward(demonstration, states):
    # Current ee, obj, goal positions
    ee_State = states[:2] # X, Y, Z
    obj_State = states[2:4] # X, Y
    goal_State = states[4:] # X, Y
    
    #initial ee, obj, goal positions
    initial_ee_pose = demonstration[0, :2]
    initial_obj_pose = demonstration[0, 2:4]
    initial_goal_pose = demonstration[0, 4:]
    
    # print(f"ee_State: {ee_State}")
    # print(f"obj_State: {obj_State}")
    # print(f"goal_State: {goal_State}")
    
    # print(f"initial_ee_pose: {initial_ee_pose}")
    # print(f"initial_obj_pose: {initial_obj_pose}")
    # print(f"initial_goal_pose: {initial_goal_pose}")
    
    ee_obj_dif = ((ee_State[0]-obj_State[0])**2+(ee_State[1]-obj_State[1])**2)
    goal_obj_dif = ((obj_State[0]-goal_State[0])**2+(obj_State[1]-goal_State[1])**2)
    initial_ee_obj_dif = ((initial_ee_pose[0]-initial_obj_pose[0])**2+(initial_ee_pose[1]-initial_obj_pose[1])**2)
    initial_goal_obj_dif = ((initial_obj_pose[0]-initial_goal_pose[0])**2+(initial_obj_pose[1]-initial_goal_pose[1])**2)
    
    return ((initial_goal_obj_dif-goal_obj_dif)/initial_goal_obj_dif)*100+((initial_ee_obj_dif-ee_obj_dif)/initial_ee_obj_dif)
    

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
    

if __name__ == "__main__":
    time_step = 300
    for i in range(num_of_demos):
        env.reset()
        
        # Closing gripper 
        env.close_gripper()
        
        loaded_demo = np.load(f"demos/demo_{i+1}.npy")
        
        demo_int=interpolate_demonstration(loaded_demo, steps=time_step)
        interpolated_demonstrations=replay_demonstration(demo_int, loaded_demo, env)
        
        #Saving interpolated demonstrations -> timestep=200
        np.save(f"demos_int/demo_{i+1}.npy", interpolated_demonstrations)
        print(f"Interpolated demonstration {i+1} saved")