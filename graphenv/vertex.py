from abc import abstractmethod
from typing import Dict, Generic, List, Optional, Sequence, TypeVar

import gym

V = TypeVar("V")


class Vertex(Generic[V]):
    """Abstract class defining a vertex in a graph. To implement a graph using
    this class, subclass Vertex and implement the abstract methods below.

    Args:
        Generic (V): The implementing vertex subclass.

    Attributes:
        _children (Optional[List]) : memoized list of child vertices
        _observation (Optional[any]) : memoized observation of this vertex
    """

    def __init__(self) -> None:
        self._children: Optional[List] = None
        self._observation: Optional[any] = None

    @property
    @abstractmethod
    def observation_space(self) -> gym.spaces.Space:
        """Gets the vertex observation space, used to define the structure of
        the data returned when observing a vertex.

        Returns:
            gym.spaces.Space: Vertex observation space
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def root(self) -> V:
        """Gets the root vertex of the graph. Not required to always return the
        same vertex.

        Returns:
            N: The root vertex of the graph.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def reward(self) -> float:
        """Gets the reward for this vertex.

        Returns:
            float: reward for this vertex
        """
        raise NotImplementedError

    @abstractmethod
    def _get_children(self) -> Sequence[V]:
        """Gets the child verticies of this vertex.

        Returns:
            Sequence[N]: Sequence of child verticies.
        """
        raise NotImplementedError

    @abstractmethod
    def _make_observation(self) -> any:
        """Gets an observation of this vertex. This observation should have
        the same shape as described by the vertex observation space.

        Returns:
            any: Observation with the same shape as defined by
                the observation space.
        """
        raise NotImplementedError

    @property
    def children(self) -> List[V]:
        """
        Gets the child verticies of this vertex.
        Acts as a wrapper that memoizes calls to _get_children() and
        ensures that it is a list. If you would like a different behavior,
        such as stochastic child verticies, override this property.

        Returns:
            List[N] : List of child verticies
        """
        if self._children is None:
            self._children = list(self._get_children())
        return self._children

    @property
    def observation(self) -> any:
        """
        Gets the observation of this vertex.
        Acts as a wrapper that memoizes calls to _make_observation().
        If you would like a different behavior,
        such as stochastic observations, override this property.

        Returns:
            Observation of this vertex.
        """
        if self._observation is None:
            self._observation = self._make_observation()
        return self._observation

    @property
    def terminal(self) -> bool:
        """
        Returns:
            True if this is a terminal vertex in the graph.
        """
        return len(self.children) == 0

    @property
    def info(self) -> Dict:
        """
        Returns:
            An optional dictionary with additional information about the state
        """
        return dict()
