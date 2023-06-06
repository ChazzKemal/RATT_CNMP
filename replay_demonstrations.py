from env_keyop_demonstration import *

if __name__ == "__main__":
    for i in range(num_of_demos):
        env.reset()
        
        loaded_demo = np.load(f"demos/demo_{i+1}.npy")
        ee_States = loaded_demo[:, :3]
        #replay_demonstration(ee_States, env)
        
        curve=interpolate_demonstration(ee_States, steps=400)
        replay_demonstration(curve, env)