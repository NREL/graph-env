import logging
import warnings
from typing import Dict, Tuple

import gym
import numpy as np
from ray.rllib.env.env_context import EnvContext

import graphenv.space_util as space_util
from graphenv.vertex import V

logger = logging.getLogger(__name__)


class GraphEnv(gym.Env):
    """
    Defines an OpenAI Gym Env for traversing a graph using the current vertex
    as the state, and the successor verticies as actions.

    GraphEnv uses composition to supply the per-vertex model of type Vertex,
    which defines the graph via it's _get_children() method.

    Attributes:
        state: current vertex
        max_num_children: maximum number of actions considered at a time
        _action_mask_key: key under which the action mask is stored in the root
            observation space dict
        _vertex_observation_key: key under which the per-action vertex observations are
            stored in the root observation space dict
    """

    state: V
    max_num_children: int
    _action_mask_key: str
    _vertex_observation_key: str

    def __init__(self, env_config: EnvContext) -> None:
        """Initializes a GraphEnv instance.

        Args:
            env_config (dict): A dictionary of parameters, required to conform with
                rllib's environment initialization. Should contain the following keys:
                state (N): Current vertex
                max_num_children (int): maximum number of children considered at a time
                action_mask_key (str, optional): key under which the action mask is
                    stored in the root observation space dict. Defaults to "action_mask"
                vertex_observation_key (str, optional): key under which the per-action
                    vertex observations are stored in the root observation space dict.
                    Defaults to "vertex_observations".
        """
        super().__init__()

        logger.debug("entering graphenv construction")
        self.state = env_config["state"]
        self.max_num_children = env_config["max_num_children"]

        try:
            self._action_mask_key = env_config["action_mask_key"]
        except KeyError:
            self._action_mask_key = "action_mask"

        try:
            self._vertex_observation_key = env_config["vertex_observation_key"]
        except KeyError:
            self._vertex_observation_key = "vertex_observations"

        num_vertex_observations = 1 + self.max_num_children
        self.observation_space = gym.spaces.Dict(
            {
                self._action_mask_key: gym.spaces.MultiBinary(num_vertex_observations),
                self._vertex_observation_key: space_util.broadcast_space(
                    self.state.observation_space, (num_vertex_observations,)
                ),
            }
        )
        self.action_space = gym.spaces.Discrete(self.max_num_children)
        logger.debug("leaving graphenv construction")

    def reset(self) -> Dict[str, np.ndarray]:
        """Reset this state to the root vertex. It is possible for state.root to
        return different root verticies on each call.

        Returns:
            Dict[str, np.ndarray]: Observation of the root vertex.
        """
        self.state = self.state.root
        return self.make_observation()

    def step(self, action: int) -> Tuple[Dict[str, np.ndarray], float, bool, dict]:
        """Steps the envirionment to a new state by taking an action. In the
        case of GraphEnv, the action specifies which next vertex to move to and
        this method advances the environment to that vertex.

        Args:
            action (int): The index of the child vertex of self.state to move to.

        Raises:
            RuntimeError: When action is an invalid index.

        Returns:
            Tuple[Dict[str, np.ndarray], float, bool, dict]: Tuple of:
                a dictionary of the new state's observation,
                the reward recieved by moving to the new state's vertex,
                a bool which is true iff the new stae is a terminal vertex,
                a dictionary of debugging information related to this call
        """

        if len(self.state.children) > self.max_num_children:
            raise RuntimeError(
                f"State {self.state} has {len(self.state.children)} children "
                f"(> {self.max_num_children})"
            )

        if action not in self.action_space:
            raise RuntimeError(
                f"Action {action} outside the action space of state {self.state}: "
                f"{len(self.state.children)} max children"
            )

        try:
            # Move the state to the next action
            self.state = self.state.children[action]
        except IndexError:
            warnings.warn(
                "Attempting to chose a masked child state. This is either due to "
                "rllib's env pre_check module, or due to a failure of the policy model "
                "to mask invalid actions. Returning the current state to satisfy the "
                "pre_check module.",
                RuntimeWarning,
            )

        result = (
            self.make_observation(),
            self.state.reward,
            self.state.terminal,
            self.state.info,
        )
        logger.debug(
            f"{type(self)}: {result[1]} {result[2]}, {result[3]},"
            f" {len(self.state.children)}"
        )
        return result

    def make_observation(self) -> Dict[str, any]:
        """
        Makes an observation for this state which includes observations of
        each possible action, and the current state.

        Expects the action observations to all be Dicts with the same keys.

        Returns a column-oriented representation, a Dict with keys matching
        the action observation keys, and values that are the current state
        and every action's values for that key concatenated into numpy arrays.

        The current state is the 0th entry in these arrays, and the children
        are offset by one index to accomodate that.

        Returns:
            Dict[str, any] : Dictionary consisting of {self._action_mask_key : bool
                action mask Numpy array, self._vertex_observation_key : stacked vertex
                observations}

        """

        num_children = 1 + self.max_num_children
        action_mask = np.zeros(num_children, dtype=bool)
        action_observations = [self.state.observation] * num_children

        for i, successor in enumerate(self.state.children):
            action_observations[i + 1] = successor.observation
            action_mask[i + 1] = True

        return {
            self._action_mask_key: action_mask,
            self._vertex_observation_key: space_util.stack_observations(
                self.observation_space[self._vertex_observation_key],
                action_observations,
            ),
        }
