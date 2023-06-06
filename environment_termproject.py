import time

import torch
import torchvision.transforms as transforms
import numpy as np

import environment_base

class TermProject_env(environment_base.BaseEnv):
    def __init__(self, obj_position, goal_position, flag_rand=True, **kwargs) -> None:
        super().__init__(**kwargs)
        self._delta = 0.05
        self._goal_thresh = 0.01
        self._max_timesteps = 50
        self._flag_rand = flag_rand
        self._predef_position = np.array([np.hstack((obj_position, 1.5)),np.hstack((goal_position, 1.025))])
        
    def _create_scene(self, seed=None):
        if self._flag_rand:
            if seed is not None:
                np.random.seed(seed)
            obj_pos =  [np.random.uniform(0.25, 0.75),
                        np.random.uniform(-0.3, 0.3),
                        1.5]
            goal_pos = [np.random.uniform(0.25, 0.75),
                        np.random.uniform(-0.3, 0.3),
                        1.025]
            scene = environment_base.create_tabletop_scene()
            #object
            environment_base.create_object(scene, "box", pos=obj_pos, quat=[0, 0, 0, 1],
                                    size=[0.03, 0.03, 0.03], rgba=[0.8, 0.2, 0.2, 1],
                                    name="obj1")
            #goal
            environment_base.create_visual(scene, "cylinder", pos=goal_pos, quat=[0, 0, 0, 1],
                                    size=[0.05, 0.005], rgba=[0.2, 1.0, 0.2, 1],
                                    name="goal")
        else:
            scene = environment_base.create_tabletop_scene()
            #object
            environment_base.create_object(scene, "box", pos=self._predef_position[0], quat=[0, 0, 0, 1],
                                    size=[0.03, 0.03, 0.03], rgba=[0.8, 0.2, 0.2, 1],
                                    name="obj1")
            #goal
            environment_base.create_visual(scene, "cylinder", pos=self._predef_position[1], quat=[0, 0, 0, 1],
                                    size=[0.05, 0.005], rgba=[0.2, 1.0, 0.2, 1],
                                    name="goal")
        return scene

    def state(self):
        if self._render_mode == "offscreen":
            self.viewer.update_scene(self.data, camera="topdown")
            pixels = torch.tensor(self.viewer.render().copy(), dtype=torch.uint8).permute(2, 0, 1)
        else:
            pixels = self.viewer.read_pixels(camid=1).copy()
            pixels = torch.tensor(pixels, dtype=torch.uint8).permute(2, 0, 1)
            pixels = transforms.functional.center_crop(pixels, min(pixels.shape[1:]))
            pixels = transforms.functional.resize(pixels, (128, 128))
        return pixels / 255.0

    def high_level_state(self):
        ee_pos = self.data.site(self._ee_site).xpos[:3]
        obj_pos = self.data.body("obj1").xpos[:2]
        goal_pos = self.data.site("goal").xpos[:2]
        return np.concatenate([ee_pos, obj_pos, goal_pos]) # ee-> 3D, obj-> 2D, goal-> 2D

    def reward(self):
        state = self.high_level_state()
        ee_pos = state[:2]
        obj_pos = state[2:4]
        goal_pos = state[4:6]
        ee_to_obj = max(100*np.linalg.norm(ee_pos - obj_pos), 1)
        obj_to_goal = max(100*np.linalg.norm(obj_pos - goal_pos), 1)
        return 1/(ee_to_obj) + 1/(obj_to_goal)

    def is_terminal(self):
        obj_pos = self.data.body("obj1").xpos[:2]
        goal_pos = self.data.site("goal").xpos[:2]
        return np.linalg.norm(obj_pos - goal_pos) < self._goal_thresh

    def is_truncated(self):
        return self._t >= self._max_timesteps

    # def step(self, action_id):
    #     action = self._actions[action_id] * self._delta
    #     ee_pos = self.data.site(self._ee_site).xpos[:2]
    #     target_pos = np.concatenate([ee_pos, [1.06]])
    #     target_pos[:2] = np.clip(target_pos[:2] + action, [0.25, -0.3], [0.75, 0.3])
    #     self._set_ee_in_cartesian(target_pos, rotation=[-90, 0, 180], n_splits=30, threshold=0.04)
    #     self._t += 1

    #     state = self.state()
    #     reward = self.reward()
    #     terminal = self.is_terminal()
    #     truncated = self.is_truncated()
    #     return state, reward, terminal, truncated
