from abc import abstractmethod
from typing import Dict, Generic, List, Optional, Sequence, TypeVar

import gym
import numpy as np

N = TypeVar("N")


class Vertex(Generic[N]):
    def __init__(
        self,
        max_num_actions: int,
    ) -> None:
        super().__init__()
        self.max_num_actions: int = max_num_actions
        self._next_actions: Optional[List] = None
        self._observation: Optional[Dict[str, np.ndarray]] = None

    @property
    @abstractmethod
    def observation_space(self) -> gym.spaces.Dict:
        raise NotImplementedError

    @property
    @abstractmethod
    def root(self) -> N:
        raise NotImplementedError

    @property
    @abstractmethod
    def reward(self) -> float:
        raise NotImplementedError

    @abstractmethod
    def _get_next_actions(self) -> Sequence[N]:
        raise NotImplementedError

    @abstractmethod
    def _make_observation(self) -> Dict[str, np.ndarray]:
        raise NotImplementedError

    @property
    def next_actions(self) -> List[N]:
        """Cache the actions from the given node"""
        if self._next_actions is None:
            self._next_actions = list(self._get_next_actions())
        return self._next_actions

    @property
    def observation(self) -> Dict[str, np.ndarray]:
        """Cache the observation construction"""
        if self._observation is None:
            self._observation = self._make_observation()
        return self._observation

    @property
    def terminal(self):
        """Whether there are any valid actions from the given node"""
        return len(self.next_actions) == 0

    @property
    def info(self) -> Dict:
        """An optional dictionary with additional information about the state"""
        return dict()
