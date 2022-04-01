from graphenv import tf


class GraphModelBellmanMixin:
    """
    Mixin for use with GraphModel that evaluates the current state as the
    max of the successor state value assesments.
    """

    def _forward_total_value(self):
        """
        Overrides state evaluation, replacing it with a Bellman backup returning
        the max over all successor states's values.

        Returns:
            Tensor of state values.
        """
        return tf.reduce_max(self.action_values, axis=1)
