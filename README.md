[![PyPI version](https://badge.fury.io/py/graphenv.svg)](https://badge.fury.io/py/graphenv)
[![Total alerts](https://img.shields.io/lgtm/alerts/g/NREL/graph-env.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/NREL/graph-env/alerts/)
[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/NREL/graph-env.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/NREL/graph-env/context:python)
[![DOI](https://zenodo.org/badge/470330187.svg)](https://zenodo.org/badge/latestdoi/470330187)
[![DOI](https://joss.theoj.org/papers/10.21105/joss.04621/status.svg)](https://doi.org/10.21105/joss.04621)


# graph-env

The `graphenv` Python library is designed to
1) make graph search problems more readily expressible as RL problems via an extension of the OpenAI gym API while
2) enabling their solution via scalable learning algorithms in the popular RLLib library.

RLLib provides out-of-the-box support for both parametrically-defined actions and masking of invalid actions. However, native support for action spaces where the action _choices_ change for each state is challenging to implement in a computationally efficient fashion. The `graphenv` library provides utility classes that simplify the flattening and masking of action observations for choosing from a set of successor states at every node in a graph search.

## Installation

Graphenv can be installed with pip:
```
pip install graphenv
```

## Quick Start

`graph-env` allows users to create a customized graph search by subclassing the `Vertex` class. Basic examples are provided in the [`graphenv/examples`](graphenv/examples) folder. The following code snippet shows how to randomly sample from valid actions for a random walk down a 1D corridor:

```python
import random
from graphenv.examples.hallway.hallway_state import HallwayState
from graphenv.graph_env import GraphEnv

state = HallwayState(corridor_length=10)
env = GraphEnv({"state": state, "max_num_children": 2})

obs = env.make_observation()
done = False
total_reward = 0

while not done:
    action = random.choice(range(len(env.state.children)))
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    total_reward += reward
```

Additional details on this example are given in the [documentation](https://nrel.github.io/graph-env/examples/hallway.html)


## Documentation

The documentation is hosted on [GitHub Pages](https://nrel.github.io/graph-env/)


## Contributing

We welcome bug reports, suggestions for new features, and pull requests. See our [contributing guidelines](CONTRIBUTING.md) for more details.

## License

`graph-env` is licensed under the [BSD 3-Clause License](LICENSE).
Copyright (c) 2022, Alliance for Sustainable Energy, LLC
