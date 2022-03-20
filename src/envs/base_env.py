from abc import abstractmethod
from dataclasses import asdict
from time import time
from pathlib import Path

from gym3 import Env
from opensim import Model, PrescribedController, Constant, Manager
import pandas as pd


class BaseOpenSimEnv(Env):
    def __init__(
        self,
        model_path: Path,
        visualize: bool = False,
        integrator_accuracy=0.001,
        delta_time=1 / 200,
    ):
        self.delta_time = delta_time
        self.visualize = visualize
        self.integrator_accuracy = integrator_accuracy
        self.current_step = 0
        self.step_info = {}

        assert (
            self.observation_space is not None
        ), "Create a subclass and set the observation_space field"
        assert (
            self.action_space is not None
        ), "Create a subclass and set the action_space field"

        self.model = Model(str(model_path))
        self.model.setUseVisualizer(self.visualize)

        self.controller = self.create_and_add_controller()

        self.state = self.model.initSystem()
        self.reset_model(pose=self.get_reset_pose())

        self.record = None

    def create_and_add_controller(self):
        """Create a controller for all actuators, and add it to the model."""
        controller = PrescribedController()

        actuator_set = self.model.getActuators()
        for j in range(actuator_set.getSize()):
            actuator = actuator_set.get(j)
            controller.addActuator(actuator)
            # print(CoordinateActuator.safeDownCast(actuator).get_coordinate())

            func = Constant(0.0)
            controller.prescribeControlForActuator(j, func)

        self.model.addController(controller)

        return controller

    def action_to_control_value(self, action, _actuator):
        return action

    def set_control_values_from_action(self, controller, action):
        """Set the actuator inputs to constants based on the action vector"""

        function_set = controller.get_ControlFunctions()
        actuator_set = self.model.getActuators()

        assert actuator_set.getSize() == function_set.getSize() == len(action), f"{str(len(action))} == {str(function_set.getSize())} == {str(actuator_set.getSize())}"

        for idx in range(actuator_set.getSize()):
            func = Constant.safeDownCast(function_set.get(idx))
            ctrl = self.action_to_control_value(action[idx], actuator_set.get(idx))
            func.setValue(float(ctrl))

    def set_pose(self, pose):
        """Called during reset, overwrite to anble reset to pose."""
        raise NotImplementedError

    def get_reset_pose(self):
        """Implement this to reset to a specific pose"""
        return None

    @abstractmethod
    def get_observation(self):
        """Called during reset and step.

        Should return the observation vector, according to 'self.observation_space'
        """
        raise NotImplementedError

    @abstractmethod
    def get_reward(self):
        """Overwrite in baseclass. Returns the reward at the current state."""
        raise NotImplementedError

    @abstractmethod
    def is_done(self):
        """Overwrite in baseclass. Retruns true, if this episode is over."""
        raise NotImplementedError

    def render(self, mode="human"):
        """Render is not supported"""
        raise NotImplementedError()

    def get_info(self):
        """Overwrite in baseclass, to return additional info for every step."""
        return self.step_info

    def reset_model(self, pose=None, start_step=0):
        """Reset without creating observation. This method might implement a reset to a specific pose."""
        self.state = self.model.initializeState()
        self.manager = None

        if pose is not None:
            self.set_pose(pose)

        self.model.equilibrateMuscles(self.state)
        self.state.setTime(start_step * self.delta_time)

        self.manager = Manager(self.model)
        self.manager.setIntegratorAccuracy(self.integrator_accuracy)
        self.manager.initialize(self.state)

        self.current_step = start_step

    def step(self, action):
        """Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.

        Accepts an action and returns a tuple (observation, reward, done, info).

        Args:
            action (object): an action provided by the agent

        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (bool): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        self.step_info = {}

        self.set_control_values_from_action(self.controller, action)

        self.current_step += 1

        self.state = self.manager.integrate(self.current_step * self.delta_time)

        observation = self.get_observation()
        reward = self.get_reward()

        if isinstance(self.record, list):
            self.record.append(self.state_as_row())

        done = self.is_done()
        info = self.get_info()
        return observation, reward, done, info

    def state_as_row(self):
        raise NotImplementedError()

    def record_next(self):
        if self.record is None:
            self.record = "on-next-reset"

    def reset(self):
        """Resets the environment to an initial state and returns an initial
        observation.

        Note that this function should not reset the environment's random
        number generator(s); random variables in the environment's state should
        be sampled independently between multiple calls to `reset()`. In other
        words, each call of `reset()` should yield an environment suitable for
        a new episode, independent of previous episodes.

        Returns:
            observation (object): the initial observation.
        """
        self.current_step = 0

        self.reset_model(pose=self.get_reset_pose())

        if self.record == "on-next-reset":
            self.record = [self.state_as_row()]
        elif isinstance(self.record, list):
            frame = pd.DataFrame([asdict(r) for r in self.record])
            frame.to_csv(f"recording_{round(time())}.csv")
            self.record = None

        return self.get_observation()
