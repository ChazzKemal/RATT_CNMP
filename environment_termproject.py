import time

import torch
import torchvision.transforms as transforms
import numpy as np

import environment_base

import joblib

class CNP(torch.nn.Module):
    def __init__(self, in_shape, hidden_size, num_hidden_layers, min_std=0.1):
        super(CNP, self).__init__()
        self.d_x = in_shape[0]
        self.d_y = in_shape[1]

        self.encoder = []
        self.encoder.append(torch.nn.Linear(self.d_x + self.d_y, hidden_size))
        self.encoder.append(torch.nn.ReLU())
        for _ in range(num_hidden_layers - 1):
            self.encoder.append(torch.nn.Linear(hidden_size, hidden_size))
            self.encoder.append(torch.nn.ReLU())
        self.encoder.append(torch.nn.Linear(hidden_size, hidden_size))
        self.encoder = torch.nn.Sequential(*self.encoder)

        self.query = []
        self.query.append(torch.nn.Linear(hidden_size + self.d_x, hidden_size))
        self.query.append(torch.nn.ReLU())
        for _ in range(num_hidden_layers - 1):
            self.query.append(torch.nn.Linear(hidden_size, hidden_size))
            self.query.append(torch.nn.ReLU())
        self.query.append(torch.nn.Linear(hidden_size, 2 * self.d_y))
        self.query = torch.nn.Sequential(*self.query)

        self.min_std = min_std

    def nll_loss(self, observation, target, target_truth, observation_mask=None, target_mask=None):
        '''
        The original negative log-likelihood loss for training CNP.
        Parameters
        ----------
        observation : torch.Tensor
            (n_batch, n_context, d_x+d_y) sized tensor that contains context
            points.
            d_x: the number of query dimensions
            d_y: the number of target dimensions.
        target : torch.Tensor
            (n_batch, n_target, d_x) sized tensor that contains query dimensions
            of target (query) points.
            d_x: the number of query dimensions.
            note: n_context and n_target does not need to be the same size.
        target_truth : torch.Tensor
            (n_batch, n_target, d_y) sized tensor that contains target
            dimensions (i.e., prediction dimensions) of target points.
            d_y: the number of target dimensions
        observation_mask : torch.Tensor
            (n_batch, n_context) sized tensor indicating which entries should be
            used in aggregation. Used for batch input.
        target_mask : torch.Tensor
            (n_batch, n_target) sized tensor indicating which entries should be
            used for loss calculation. Used for batch input.
        Returns
        -------
        loss : torch.Tensor (float)
            The NLL loss.
        '''
        mean, std = self.forward(observation, target, observation_mask)
        dist = torch.distributions.Normal(mean, std)
        nll = -dist.log_prob(target_truth)
        if target_mask is not None:
            # sum over the sequence (i.e. targets in the sequence)
            nll_masked = (nll * target_mask.unsqueeze(2)).sum(dim=1)
            # compute the number of entries for each batch entry
            nll_norm = target_mask.sum(dim=1).unsqueeze(1)
            # first normalize, then take an average over the batch and dimensions
            loss = (nll_masked / nll_norm).mean()
        else:
            loss = nll.mean()
        return loss

    def forward(self, observation, target, observation_mask=None):
        '''
        Forward pass of CNP.
        Parameters
        ----------
        observation : torch.Tensor
            (n_batch, n_context, d_x+d_y) sized tensor where d_x is the number
            of the query dimensions, d_y is the number of target dimensions.
        target : torch.Tensor
            (n_batch, n_target, d_x) sized tensor where d_x is the number of
            query dimensions. n_context and n_target does not need to be the
            same size.
        observation_mask : torch.Tensor
            (n_batch, n_context) sized tensor indicating which entries should be
            used in aggregation.
        Returns
        -------
        mean : torch.Tensor
            (n_batch, n_target, d_y) sized tensor containing the mean
            prediction.
        std : torch.Tensor
            (n_batch, n_target, d_y) sized tensor containing the standard
            deviation prediction.
        '''
        h = self.encode(observation)
        r = self.aggregate(h, observation_mask=observation_mask)
        h_cat = self.concatenate(r, target)
        query_out = self.decode(h_cat)
        mean = query_out[..., :self.d_y]
        logstd = query_out[..., self.d_y:]
        std = torch.nn.functional.softplus(logstd) + self.min_std
        return mean, std

    def encode(self, observation):
        h = self.encoder(observation)
        return h

    def decode(self, h):
        o = self.query(h)
        return o

    def aggregate(self, h, observation_mask):
        # this operation is equivalent to taking mean but for
        # batched input with arbitrary lengths at each entry
        # the output should have (batch_size, dim) shape

        if observation_mask is not None:
            h = (h * observation_mask.unsqueeze(2)).sum(dim=1)  # mask unrelated entries and sum
            normalizer = observation_mask.sum(dim=1).unsqueeze(1)  # compute the number of entries for each batch entry
            r = h / normalizer  # normalize
        else:
            # if observation mask is none, we assume that all entries
            # in the batch has the same length
            r = h.mean(dim=1)
        return r

    def concatenate(self, r, target):
        num_target_points = target.shape[1]
        r = r.unsqueeze(1).repeat(1, num_target_points, 1)  # repeating the same r_avg for each target
        h_cat = torch.cat([r, target], dim=-1)
        return h_cat


class TermProject_env(environment_base.BaseEnv):
    def __init__(self, origin, obj_position, goal_position, flag_rand=True, density=1000, **kwargs) -> None:
        super().__init__(**kwargs)
        self._delta = 0.05
        self._goal_thresh = 0.01
        self._max_timesteps = 50
        self._flag_rand = flag_rand
        self._density_object = density
        self._predef_position = np.array([np.hstack((obj_position, 1.5)),np.hstack((goal_position, 1.025))])
        self._origin = origin
        self.go_origin()
        self.close_gripper()
        # self.object_start_pos# Abdullah 07.06.23
        # self.goal_pos# Abdullah 07.06.23
        #self.set_initial_pos(obj_start_pos,goal_pos) # Abdullah 07.06.23
        
        
    # def set_initial_pos(self,obj_start_pos,goal_pos):# Abdullah 07.06.23
    #     self.object_start_pos = obj_start_pos
    #     self.goal_pos = goal_pos


    # def _create_scene(self, seed=None):# Abdullah 07.06.23
    #     if seed is not None:
    #         np.random.seed(seed)
    #     scene = environment.create_tabletop_scene()
    #     # height = np.random.uniform(0.03, 0.1)
    #     environment.create_object(scene, "box", pos=self.object_start_pos, quat=[0, 0, 0, 1],
    #                               size=[0.03, 0.03, 0.03], rgba=[0.8, 0.2, 0.2, 1],
    #                               name="obj1")
        
        
    #     environment.create_object(scene, "box", pos=self.goal_pos, quat=[0, 0, 0, 1],
    #                               size=[0.03, 0.03, 0.03], rgba=[0.8, 0.8, 0.2, 1],
    #                               name="obj2")
    #     return scene
        
        
    def go_origin(self):
        # Setting initial position; i.e., origin point
        self._set_ee_in_cartesian(self._origin, rotation=[-90, 0, 180], n_splits=100, max_iters=100, threshold=0.05)
        
        
    def close_gripper(self):
        # Closing the gripper
        joint_poses = self._get_joint_position()  #"ur5e/robotiq_2f85/right_driver_joint"
        joint_poses[6]=np.pi/6 # setting gripper joint to close
        self._set_joint_position(joint_poses)
        
        
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
                                    name="obj1", density=self._density_object)
            #goal
            environment_base.create_visual(scene, "cylinder", pos=goal_pos, quat=[0, 0, 0, 1],
                                    size=[0.05, 0.005], rgba=[0.2, 1.0, 0.2, 1],
                                    name="goal")
        else:
            scene = environment_base.create_tabletop_scene()
            #object
            environment_base.create_object(scene, "box", pos=self._predef_position[0], quat=[0, 0, 0, 1],
                                    size=[0.03, 0.03, 0.03], rgba=[0.8, 0.2, 0.2, 1],
                                    name="obj1", density=self._density_object)
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
        ee_pos = self.data.site(self._ee_site).xpos[:2]
        obj_pos = self.data.body("obj1").xpos[:2]
        goal_pos = self.data.site("goal").xpos[:2]
        return np.concatenate([ee_pos, obj_pos, goal_pos]) # ee-> 2D, obj-> 2D, goal-> 2D

    def reward(self):
        state = self.high_level_state()
        ee_pos = state[:2] # 3rd index is height
        obj_pos = state[3:5]
        goal_pos = state[5:7]
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


# if __name__ == "__main__":
#     env = Hw5Env(render_mode="gui")
#     states_arr = []
#     for i in range(1000):
#         env.reset()
#         p_1 = np.array([0.5, 0.3, 1.04])
#         p_2 = np.array([0.5, 0.15, np.random.uniform(1.04, 1.4)])
#         p_3 = np.array([0.5, -0.15, np.random.uniform(1.04, 1.4)])
#         p_4 = np.array([0.5, -0.3, 1.04])
#         points = np.stack([p_1, p_2, p_3, p_4], axis=0)
#         curve = bezier(points)
#         # curve = np.array([[np.sin(2*np.pi*i/100)/2+.5,np.cos(2*np.pi*i/100)/2+.5,1.06] for i in range(100)])
#         # curve = np.array([[0.5,i*0.0001,1.04] for i in range(100)])
#         curve = np.array([[.5+np.sin(i/100*2*np.pi)*.2,np.cos(i/100*2*np.pi)*.2,1.1] for i in range(500)])
#         env._set_ee_in_cartesian(curve[0], rotation=[-90, 0, 180], n_splits=100, max_iters=100, threshold=0.05)
#         states = []
#         for p in curve:
#             env._set_ee_pose(p, rotation=[-90, 0, 180], max_iters=10)
#             states.append(env.high_level_state())
#         states = np.stack(states)
#         states_arr.append(states)
#         print(f"Collected {i+1} trajectories.", end="\r")
#         # joblib.dump(states_arr,"hw5data.pkl")

#     fig, ax = plt.subplots(1, 2)
#     for states in states_arr:
#         ax[0].plot(states[:, 0], states[:, 1], alpha=0.2, color="b")
#         ax[0].set_xlabel("e_y")
#         ax[0].set_ylabel("e_z")
#         ax[1].plot(states[:, 2], states[:, 3], alpha=0.2, color="r")
#         ax[1].set_xlabel("o_y")
#         ax[1].set_ylabel("o_z")
#     plt.show()
