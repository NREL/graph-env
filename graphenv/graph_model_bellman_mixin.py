from ray.rllib.utils.framework import try_import_tf

tf1, tf, tfv = try_import_tf()


class GraphModelBellmanMixin:
    """
    Mixing for use with GraphModel that evaluates the current state as the
    max of the successor state value assesments.
    """

    def _forward_total_value(self):
        """
        Overrides state evaluation, replacing it with a Bellman backup returning
        the max over all successor states's values.
        """
        return tf.reduce_max(self.action_values, axis=1)
