import functools
import os
import warnings
from typing import Optional, Callable, Union

import gymnasium
import numpy as np
import pygame
from highway_env import utils
from highway_env.envs import IntersectionEnv
from highway_env.envs.common.action import ActionType
from highway_env.utils import Vector
from highway_env.vehicle.controller import MDPVehicle
from highway_env.vehicle.kinematics import Vehicle

warnings.simplefilter(action='ignore', category=FutureWarning)


def calc_distance(a, b):
    return np.sqrt(sum(((a[0] - b[0]) ** 2, (a[1] - b[1]) ** 2)))


class EgoVehicle(MDPVehicle):
    """A vehicle controlled by the agent."""

    def speed_control(self, target_speed: float) -> float:
        return target_speed

    def act(self, action: Union[dict, str] = None) -> None:
        """
        Perform a high-level action.

        - If the action is a speed change, choose speed from the allowed discrete range.
        - Else, forward action to the ControlledVehicle handler.

        :param action: a high-level action
        """
        if action is not None:
            self.target_speed = action if action + self.speed > 0 else min(-self.speed, 0)

        super().act()


class DiscreteMeta(ActionType):
    """
    """

    def __init__(self,
                 env: 'AbstractEnv',
                 target_speeds: Optional[Vector] = None,
                 **kwargs) -> None:
        """
        Create a discrete action space of meta-actions.

        :param env: the environment
        :param longitudinal: include longitudinal actions
        :param lateral: include lateral actions
        :param target_speeds: the list of speeds the vehicle is able to track
        """
        super().__init__(env)
        self.target_speeds = np.array(target_speeds) if target_speeds is not None else MDPVehicle.DEFAULT_TARGET_SPEEDS

    def space(self) -> gymnasium.spaces.Space:
        return gymnasium.spaces.Discrete(3)

    @property
    def vehicle_class(self) -> Callable:
        return functools.partial(EgoVehicle, target_speeds=self.target_speeds)

    def act(self, action: int) -> None:
        self.controlled_vehicle.act(action)


class CumulantIntersectionEnv(IntersectionEnv):
    def __init__(self, env_config, **kwargs):
        self.default_w = np.asarray(env_config['default_w']) if env_config.get('default_w', False) else np.ones((3,))
        self.observation_space_n = len(env_config["observation"]["features"])
        self.video_history = [] if env_config.get("video", False) else None
        self.flatten_obs = env_config["observation"]["flatten"]
        super(CumulantIntersectionEnv, self).__init__(config=env_config)
        self.seed_ = 42
        self.seed(self.seed_)
        self.rendering = False if self.config["offscreen_rendering"] else True
        setattr(self.observation_space, 'n', self.observation_space_n)
        setattr(self.action_space, 'n', 3)
        self.dt = 1 / env_config["policy_frequency"]
        # self.observation_space = gymnasium.spaces.Box(low=-np.inf, high=np.inf, shape=(self.observation_space_n,))
        self.observation_space = gymnasium.spaces.Box(low=-np.inf, high=np.inf,
                                                      shape=(np.multiply(*self.observation_space.shape),))

    def _get_cumulants(self, vehicle: Vehicle):
        scaled_speed = utils.lmap(self.vehicle.speed, self.config["reward_speed_range"], [0, 1])
        collision_reward = self.config["collision_reward"] * vehicle.crashed
        high_speed_reward = self.config["high_speed_reward"] * np.clip(scaled_speed, 0, 1)
        arrived_reward = self.config["arrived_reward"] if self.has_arrived(vehicle) else 0.
        cumulants = np.array([collision_reward, high_speed_reward, arrived_reward], dtype=np.float32)
        return cumulants

    def _agent_reward(self, action: int, vehicle: Vehicle) -> float:
        cumulants = self._get_cumulants(vehicle)
        reward = sum(self.default_w * cumulants)
        return reward

    def step(self, action: int):
        obs, reward, done, truncated, info = super().step(action)
        if self.video_history is not None:
            self.video_history.append(self.render('rgb_array'))
        if self.rendering:
            self.render()
        if self.flatten_obs:
            obs = obs.flatten()
        cumulants = self._get_cumulants(self.vehicle)
        info["cumulants"] = cumulants
        info["cause"] = "slow" if (done or truncated) and not self.vehicle.crashed and not self.has_arrived(
            self.vehicle) else "collision" if self.vehicle.crashed else None
        info["cost"] = abs(cumulants[0] * self.default_w[0])

        return obs, reward, done, truncated, info

    def get_max_reward(self, temp_reward):
        return np.array([1, 1, 1])

    def set_render(self, mode, save_path=None):
        pass

    def display_text_on_gui(self, name, text=None, loc=None, rel_to_ego=True):
        if self.viewer is not None:
            font1 = pygame.font.SysFont(None, 20)
            for i in range(0, len(text) + 1, 5):
                img1 = font1.render(str([f"{j:1.3f}" for j in text[i:i + 5]]), True, (0, 0, 0))
                self.viewer.screen.blit(img1, (loc[0], loc[1] + i * 5))
            pygame.display.update()
        else:
            pass

    def stop(self):
        pass

    def save_episode(self, path, video_name="video.avi",
                     frame_rate=10, scale_percent=1):
        if self.video_history is not None:
            if not os.path.exists(path):
                os.makedirs(path)
            img = self.video_history[0]
            width = int(img.shape[1] * scale_percent)
            height = int(img.shape[0] * scale_percent)
            dim = (width, height)
            # save_images_to_video(self.video_history, path, video_name, frame_rate, dim)

    def seed(self, seed: int = None):
        if isinstance(seed, (np.random.RandomState,)):
            self.np_random = seed
            return None
        else:
            seed_ = super().reset(seed=seed)
            self.seed_ = self.np_random
        return seed_

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset()
        if self.flatten_obs:
            obs = obs.flatten()
        if self.video_history is not None:
            self.video_history = []
        return obs, info
