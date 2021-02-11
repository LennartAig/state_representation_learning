"""An observation wrapper that augments observations by pixel values."""

import collections
import copy

import numpy as np
import cv2

from gym import spaces
from gym import Wrapper, ObservationWrapper


class GymWrapper(Wrapper):
    """Augment observations by pixel values."""

    def __init__(
        self,
        env,
        obs_type="pixels",
        height=64,
        width=64,
        channel_first=True,
        render_kwargs=None,
        action_repeat=4,
        action_norm=False,
    ):
        """Initializes a new pixel Wrapper.
        Args:
        ------
            env: The environment to wrap.
            render_kwargs: Optional `dict` containing keyword arguments passed
                to the `self.render` method.
            pixel_keys: Optional custom string specifying the pixel
                observation's key in the `OrderedDict` of observations.
                Defaults to 'pixels'.
        Raises:
        ------
            ValueError: If `env`'s observation spec is not compatible with the
                wrapper. Supported formats are a single array, or a dict of
                arrays.
        """

        super(GymWrapper, self).__init__(env)

        if render_kwargs is None:
            self._render_kwargs = dict(
                width=height,
                height=height,
                depth=False,
                camera_name="track",
                mode="rgb_array",
            )
        else:
            self._render_kwargs = render_kwargs
        wrapped_observation_space = env.observation_space
        self.action_repeat = action_repeat
        self.action_norm = action_norm

        if action_norm:
            assert action_norm and isinstance(
                env.action_space, spaces.Box
            ), "Action norm set. Expected Box action space, got {}".format(
                type(env.action_space)
            )
            self.a = (
                np.zeros(env.action_space.shape, dtype=env.action_space.dtype) - 1.0
            )
            self.b = (
                np.zeros(env.action_space.shape, dtype=env.action_space.dtype) + 1.0
            )
            self.action_space = spaces.Box(
                low=-1.0,
                high=1.0,
                shape=env.action_space.shape,
                dtype=env.action_space.dtype,
            )

        if obs_type == "state":
            self.obs_type = "state"
            if self.env.spec.id == "CarRacing-v0":
                raise Exception("This Environment has no true state, only pixels")
            self.observation_space = env.observation_space
        elif obs_type == "pixels":
            self.obs_type = "pixels"
            # Excpetion for car rycing
            if self.env.spec.id == "CarRacing-v0":
                pixels = self.env.observation_space
            else:
                # from mujoco_py import GlfwContext
                # GlfwContext(offscreen=True)
                pixels = self.env.render(mode="rgb_array")
            if np.issubdtype(pixels.dtype, np.integer):
                low, high = (0, 255)
                space_type = np.uint8
            elif np.issubdtype(pixels.dtype, np.float):
                low, high = (-float("inf"), float("inf"))
                space_type = np.float32
            if channel_first:
                image_shape = (
                    3,
                    self._render_kwargs["height"],
                    self._render_kwargs["width"],
                )
            else:
                image_shape = (
                    self._render_kwargs["height"],
                    self._render_kwargs["width"],
                    3,
                )
            image_space = spaces.Box(low, high, shape=image_shape, dtype=space_type)
            self.observation_space = image_space
        else:
            raise NotImplementedError("Check the Wrapper for detail")

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        return self.observation(observation)

    def step(self, action):
        done = False
        total_reward = 0
        current_step = 0
        while current_step < (self.action_repeat + 1) and not done:
            # if self.action_norm: action = self.action(action)
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            current_step += 1
        return self.observation(obs), total_reward, done, info

    def observation(self, observation):
        shape = (self._render_kwargs["height"], self._render_kwargs["width"])
        if self.obs_type == "pixels":
            observation = self.env.render(mode=self._render_kwargs["mode"])
            observation = cv2.resize(observation, shape, interpolation=cv2.INTER_AREA)
            observation = np.transpose(observation, [2, 0, 1]).copy()
        return observation

    def action(self, action):
        assert np.all(np.greater_equal(action, self.a)), (action, self.a)
        assert np.all(np.less_equal(action, self.b)), (action, self.b)
        low = self.env.action_space.low
        high = self.env.action_space.high
        action = low + (high - low) * ((action - self.a) / (self.b - self.a))
        action = np.clip(action, low, high)
        return action

    """
    def step(self, action):
        return self.env.step(self.action(action))

    def action(self, action):
        raise NotImplementedError

    def reverse_action(self, action):
        raise NotImplementedError


    def __init__(self, env):
            super(SkipWrapper, self).__init__(env)
            self.repeat_count = repeat_count
            self.stepcount = 0

        def _step(self, action):
            done = False
            total_reward = 0
            current_step = 0
            while current_step < (self.repeat_count + 1) and not done:
                self.stepcount += 1
                obs, reward, done, info = self.env.step(action)
                total_reward += reward
                current_step += 1
            if 'skip.stepcount' in info:
                raise gym.error.Error('Key "skip.stepcount" already in info. Make sure you are not stacking ' \
                                      'the SkipWrapper wrappers.')
            info['skip.stepcount'] = self.stepcount
            return obs, total_reward, done, info

        def _reset(self):
            self.stepcount = 0
            return self.env.reset()
class NormalizedActions(gym.ActionWrapper):
    def _action(self, action):
        low  = self.action_space.low
        high = self.action_space.high
        
        action = low + (action + 1.0) * 0.5 * (high - low)
        action = np.clip(action, low, high)
        
        return action

    def _reverse_action(self, action):
        low  = self.action_space.low
        high = self.action_space.high
        
        action = 2 * (action - low) / (high - low) - 1
        action = np.clip(action, low, high)
        
        return action


    """


class FilterObservation(ObservationWrapper):
    """Filter dictionary observations by their keys.
    
    Args:
        env: The environment to wrap.
        filter_keys: List of keys to be included in the observations.
    Raises:
        ValueError: If observation keys in not instance of None or
            iterable.
        ValueError: If any of the `filter_keys` are not included in
            the original `env`'s observation space
    
    """

    def __init__(self, env, filter_keys=None):
        super(FilterObservation, self).__init__(env)

        wrapped_observation_space = env.observation_space

        observation_keys = wrapped_observation_space.spaces.keys()

        if filter_keys is None:
            filter_keys = tuple(observation_keys)

        missing_keys = set(key for key in filter_keys if key not in observation_keys)

        if missing_keys:
            raise ValueError(
                "All the filter_keys must be included in the "
                "original obsrevation space.\n"
                "Filter keys: {filter_keys}\n"
                "Observation keys: {observation_keys}\n"
                "Missing keys: {missing_keys}".format(
                    filter_keys=filter_keys,
                    observation_keys=observation_keys,
                    missing_keys=missing_keys,
                )
            )
        total_size = 0
        high = 0
        low = 0
        dtype = None
        for name, space in wrapped_observation_space.spaces.items():
            high = space.high[0]
            low = space.low[0]
            total_size += space.shape[0]
            dtype = space.dtype

        self.observation_space = spaces.Box(low, high, shape=(total_size,), dtype=dtype)

        self._env = env
        self._filter_keys = tuple(filter_keys)

    def observation(self, observation):
        filter_observation = self._filter_observation(observation)
        return filter_observation

    def _filter_observation(self, observation):
        value_temp = []
        for name, value in observation.items():
            if name in self._filter_keys:
                value_temp.extend(value)
        value_temp = np.array(value_temp)
        return value_temp
