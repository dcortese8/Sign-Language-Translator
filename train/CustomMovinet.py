import tensorflow as tf

# This custom training step ignores the updated states during training as they are only important during inference.
class CustomModel(tf.keras.Model):

    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        if len(data) == 3:
            x, y, sample_weight = data
        else:
            sample_weight = None
            x, y = data
            # print(y)
            # print(x)
        with tf.GradientTape() as tape:
            pred, states = self(x, training=True)  # Forward pass

            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = self.compiled_loss(y, pred, regularization_losses=self.losses, sample_weight=sample_weight)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, pred, sample_weight=sample_weight)

        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}
