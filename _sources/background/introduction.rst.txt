.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Background

   self


The ``graphenv`` Python library is designed to

1. make graph search problems more readily expressible as RL problems via an extension
   of the OpenAI gym API while
2. enabling their solution via scalable learning algorithms in the popular RLLib
   library.

RLLib provides out-of-the-box support for both parametrically-defined actions and
masking of invalid actions. However, native support for action spaces where the action
_choices_ change for each state is challenging to implement in a computationally
efficient fashion. The `graphenv` library provides utility classes that simplify the
flattening and masking of action observations for choosing from a set of successor
states at every node in a graph search.

The intended audience for the ``graphenv`` library consist of researchers working on 
graph search problems that are amenable to a reinforcement learning formulation, 
broadly described as "learning to optimize".  This includes those working on 
classical combinatorial optimization problems such as the Traveling Salesperson 
Problem, as well as problems that do not have a clear algebraic expression but where 
the environment dynamics can be simulated, for instance, molecular design.


Installation
============

Graphenv can be installed with pip:

.. code-block::
    
    pip install graphenv

Graphenv depends on `gym`, `networkx`, `ray[tune,rllib]`, as well as one of either
`tensorflow` or `pytorch`. You can install Graphenv together with the chosen ML
framework using

.. code-block::
    
    pip install graphenv[tensorflow]

or 

.. code-block::
    
    pip install graphenv[torch]


Quick Start
===========

`graph-env` allows users to create a customized graph search by subclassing the `Vertex`
class. Basic examples are provided in the [`graphenv/examples`](graphenv/examples)
folder. The following code snippet shows how to randomly sample from valid actions for a
random walk down a 1D corridor:

.. code-block:: python

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
