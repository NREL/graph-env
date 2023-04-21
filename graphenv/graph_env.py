import logging
import warnings
from typing import Any, Dict, List, Optional, Tuple
import inspect

import gymnasium as gym
import numpy as np
from ray.rllib.env.env_context import EnvContext
from ray.rllib.utils.spaces.repeated import Repeated

from graphenv.vertex import V

logger = logging.getLogger(__name__)


class GraphEnv(gym.Env):
    """
    Defines an OpenAI Gym Env for traversing a graph using the current vertex
    as the state, and the successor vertices as actions.

    GraphEnv uses composition to supply the per-vertex model of type Vertex, which
    defines the graph via it's `_get_children()` method.

    The `env_config` dictionary should contain the following keys::

        state (N): Current vertex
        max_num_children (int): maximum number of children considered at a time.

    Args:
        env_config (dict): A dictionary of parameters, required to conform with
            rllib's environment initialization.
    """

    #: graphenv.vertex.Vertex: current vertex
    state: V

    #: int: maximum number of actions considered at a time
    max_num_children: int

    #: the observation space of the graph environment
    observation_space: gym.Space

    #: the action space, a Discrete space over `max_num_children`
    action_space: gym.Space

    # For environment rendering
    metadata: Dict[str, Any] = {"render_modes": ["human", None]}
    render_mode: Optional[str] = None

    def __init__(self, env_config: EnvContext) -> None:
        super().__init__()

        logger.debug("entering graphenv construction")
        self.state = env_config["state"]
        self.max_num_children = env_config["max_num_children"]

        num_vertex_observations = 1 + self.max_num_children
        self.observation_space = Repeated(
            self.state.observation_space, num_vertex_observations
        )

        self.action_space = gym.spaces.Discrete(self.max_num_children)
        logger.debug("leaving graphenv construction")

    # RLlib 2.3.1 does not yet support setting the 'seed' here. Using kwargs quiets the warning.
    # "Seeding will take place using 'env.seed()' and the info dict will not be returned from reset."
    #def reset(self, *, seed=None, options=None) -> Tuple[Dict[str, np.ndarray], Dict]:
    def reset(self, **kwargs) -> Tuple[Dict[str, np.ndarray], Dict]:
        """Reset this state to the root vertex. It is possible for state.root to
        return different root vertices on each call.

        Returns:
            Dict[str, np.ndarray]: Observation of the root vertex.
        """
        self.state = self.state.root
        return self.make_observation(), self.state.info

    def step(self, action: int) -> Tuple[Dict[str, np.ndarray], float, bool, bool, dict]:
        """Steps the environment to a new state by taking an action. In the
        case of GraphEnv, the action specifies which next vertex to move to and
        this method advances the environment to that vertex.

        Args:
            action (int): The index of the child vertex of self.state to move to.

        Raises:
            RuntimeError: When action is an invalid index.

        Returns:
            Tuple[Dict[str, np.ndarray], float, bool, dict]: Tuple of:
                a dictionary of the new state's observation,
                the reward received by moving to the new state's vertex,
                a bool which is true iff the new state is a terminal vertex,
                a bool which is true if the search is truncated
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
            # Skip this warning message if the call
            # came from rllib's precheck function
            # https://github.com/ray-project/ray/blob/e6dad0b961b5e962f6dc4986947ccac2d2e032cd/rllib/utils/pre_checks/env.py#L220
            skip_warning = False
            for stack_func_info in inspect.stack():
                caller_name = stack_func_info[3]
                if caller_name == "check_gym_environments":
                    skip_warning = True
            if not skip_warning:
                warnings.warn(
                    "Attempting to choose a masked child state. This is either due to "
                    "rllib's env pre_check module, or due to a failure of the policy model "
                    "to mask invalid actions. Returning the current state to satisfy the "
                    "pre_check module.",
                    RuntimeWarning,
                )

        # In RLlib 2.3, the config options "no_done_at_end", "horizon", and "soft_horizon" are no longer supported
        # according to the migration guide https://docs.google.com/document/d/1lxYK1dI5s0Wo_jmB6V6XiP-_aEBsXDykXkD1AXRase4/edit#
        # Instead, wrap your gymnasium environment with a TimeLimit wrapper, 
        # which will set truncated according to the number of timesteps 
        # see https://gymnasium.farama.org/api/wrappers/misc_wrappers/#gymnasium.wrappers.TimeLimit
        truncated = False
        result = (
            self.make_observation(),
            self.state.reward,
            self.state.terminal,
            truncated,
            self.state.info,
        )
        logger.debug(
            f"{type(self)}: {result[1]} {result[2]}, {result[3]},"
            f" {len(self.state.children)}"
        )
        return result

    def make_observation(self) -> List[any]:
        """
        Makes an observation for this state which includes observations of
        each possible action, and the current state.

        Expects the action observations to all be Dicts with the same keys.

        Returns a column-oriented representation, a Dict with keys matching
        the action observation keys, and values that are the current state
        and every action's values for that key concatenated into numpy arrays.

        The current state is the 0th entry in these arrays, and the children
        are offset by one index to accommodate that.

        Returns:

            List[any]: A list of next state observations.

        """

        assert (
            len(self.state.children) <= self.max_num_children
        ), f"{self.state} exceeds the maximum number of children"

        return [state.observation for state in (self.state, *self.state.children)]

    def render(self, mode: str = "human") -> Any:
        """Delegates to Vertex.render()"""

        if mode == "human":
            return self.state.render()
