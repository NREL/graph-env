Running ``GraphEnv`` with ``ray.tune``
============

Practical reinforcement learning will typically leverage the ``ray.tune`` infrastructure
to scale up environment rollouts and policy model training. For the hallway example, an 
example ``tensorflow`` implementation consists of the following:

.. literalinclude:: hallway_test_tf.py
    :language: python
    :linenos:

In lines 7-20, we specify configuration options for PPO, including matching the 
framework with that used in the provided ``HallwayModel`` policy. This script runs 5 
iterations of the PPO training algorithm, and the results can be monitored with 
tensorboard. 

Running the same experiment with ``pytorch`` requires writing a pytorch-compatible
policy model, demonstrated in ``graphenv.examples.hallway.hallway_model_torch``. Beyond 
this, the only required modifications to the training script to use pytorch instead of 
tensorflow are shown below:

.. literalinclude:: hallway_test_torch.py
    :language: python
    :emphasize-lines: 2,14,17
    :linenos: