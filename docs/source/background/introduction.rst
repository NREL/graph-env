.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Background

   self


The ``graphenv`` Python library is designed to

1. make graph search problems more readily expressible as RL problems via an extension of the OpenAI gym API while
2. enabling their solution via scalable learning algorithms in the popular RLLib library.

RLLib provides out-of-the-box support for both parametrically-defined actions and masking of invalid actions. However, native support for action spaces where the action _choices_ change for each state is challenging to implement in a computationally efficient fashion. The `graphenv` library provides utility classes that simplify the flattening and masking of action observations for choosing from a set of successor states at every node in a graph search.


Installation
============

Graphenv can be installed with pip:

.. code-block::
    
    pip install graphenv
