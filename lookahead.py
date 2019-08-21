#! -*- coding: utf-8 -*-

from keras import backend as K


class Lookahead(object):
    """Add the [Lookahead Optimizer](https://arxiv.org/abs/1907.08610) functionality for [keras](https://keras.io/).
    """

    def __init__(self, k=5, alpha=0.5):
        self.k = k
        self.alpha = alpha
        self.count = 0

    def inject(self, model):
        """Inject the Lookahead algorithm for the given model.
        The following code is modified from keras's _make_train_function method.
        See: https://github.com/keras-team/keras/blob/master/keras/engine/training.py#L497
        """
        if not hasattr(self.keras_model, 'train_function'):
            raise RuntimeError('You must compile your model before using it.')
        if self.keras_model.train_function is None:
            inputs = (
                self.keras_model._feed_inputs + self.keras_model._feed_targets
                + self.keras_model._feed_sample_weights)

            if (self.keras_model.uses_learning_phase
                    and not isinstance(K.learning_phase(), int)):
                inputs += [K.learning_phase()]
            fast_params = self.keras_model._collected_trainable_weights
            with K.name_scope('training'):
                with K.name_scope(
                        self.keras_model.optimizer.__class__.__name__):
                    training_updates = self.keras_model.optimizer.get_updates(
                        params=fast_params, loss=self.keras_model.total_loss)
                    slow_params = [K.variable(p) for p in fast_params]
                fast_updates = (self.keras_model.updates + training_updates)
                slow_updates, copy_updates = [], []
                for p, q in zip(fast_params, slow_params):
                    slow_updates.append(K.update(q, q + alpha * (p - q)))
                    copy_updates.append(K.update(p, q))
                # Gets loss and metrics. Updates weights at each call.
                fast_train_function = K.function(
                    inputs, [self.keras_model.total_loss] +
                    self.keras_model.metrics_tensors,
                    updates=fast_updates,
                    name='fast_train_function',
                    **self.keras_model._function_kwargs)

                def F(inputs):
                    self.count += 1
                    R = fast_train_function(inputs)
                    if self.count % count_max == 0:
                        K.batch_get_value(slow_updates)
                        K.batch_get_value(copy_updates)
                    return R

                self.keras_model.train_function = F
