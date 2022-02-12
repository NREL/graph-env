from abc import abstractmethod
from typing import Dict, Generic, List, Optional, Sequence, TypeVar

import gym
import numpy as np

N = TypeVar("N")


class Node(Generic[N]):
    def __init__(
        self,
        max_num_actions: int,
    ) -> None:

        super().__init__()
        self.max_num_actions: int = max_num_actions
        self._next_actions: Optional[List] = None
        self._observation: Optional[Dict[str, np.ndarray]] = None

    @abstractmethod
    def get_next_actions(self) -> Sequence[N]:
        pass

    @abstractmethod
    def make_observation(self) -> Dict[str, np.ndarray]:
        pass

    @property
    @abstractmethod
    def null_observation(self) -> Dict[str, np.ndarray]:
        pass

    @property
    @abstractmethod
    def observation_space(self) -> gym.spaces.Dict:
        pass

    @abstractmethod
    def get_root(self) -> N:
        pass

    @abstractmethod
    def reward(self) -> float:
        pass

    @property
    def action_space(self) -> gym.spaces.Discrete:
        return gym.spaces.Discrete(self.max_num_actions)

    @property
    def next_actions(self) -> List[N]:
        """Cache the actions from the given node"""
        if self._next_actions is None:
            self._next_actions = list(self.get_next_actions())
        return self._next_actions

    @property
    def observation(self) -> Dict[str, np.ndarray]:
        """Cache the observation construction"""
        if self._observation is None:
            self._observation = self.make_observation()
        return self._observation

    @property
    def terminal(self):
        """Whether there are any valid actions from the given node"""
        return len(self.next_actions) == 0

    @property
    def info(self) -> Dict:
        """An optional dictionary with additional information about the state"""
        return dict()
