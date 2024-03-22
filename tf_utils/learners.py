import tensorflow as tf

class FGM(tf.keras.Model):
    """
    **Fast Gradient Method for Adversarial Training**

    This class implements the Fast Gradient Method (FGM) as a Keras model for adversarial training.
    FGM perturbs the input embedding to make the model more robust to adversarial examples.
    """
    def __init__(self,
                 *args,
                 delta=0.2,
                 eps=1e-4,
                 start_step=0,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.delta = delta              # Perturbation magnitude
        self.eps = eps                  # Small constant for numerical stability
        self.start_step = start_step    # Step at which to start applying FGM
        
    def train_step_fgm(self, data):
        """
        Performs a single training step with FGM perturbation.

        """

        # source: https://stackoverflow.com/questions/62786227/how-to-track-weights-and-gradients-in-a-keras-custom-training-loo
        x, y = data # Unpack the data. Its structure depends on your model and on what you pass to `fit()`.

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        # Calculate adversarial perturbation for the embedding
        embedding = self.trainable_variables[0]
        embedding_gradients = tape.gradient(loss, [self.trainable_variables[0]])[0] # Assume first trainable variable is the embedding
        embedding_gradients = tf.zeros_like (embedding) + embedding_gradients
        delta = tf.math.divide_no_nan(self.delta * embedding_gradients , tf.math.sqrt(tf.reduce_sum(embedding_gradients**2)) + self.eps)

        # Apply adversarial perturbation and calculate model updates
        self.trainable_variables[0].assign_add(delta)
        with tf.GradientTape() as tape2:
            y_pred = self(x, training=True)
            new_loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses) # Calculate loss
            # Handle potential scaling for optimizers like Lookahead:
            if hasattr(self.optimizer, 'get_scaled_loss'):
                new_loss = self.optimizer.get_scaled_loss(new_loss)

        gradients = tape2.gradient(new_loss, self.trainable_variables) # Calculate gradients
        # Handle potential unscaling for optimizers like Lookahead
        gradients = self.optimizer.get_unscaled_gradients(gradients) if hasattr(self.optimizer, 'get_unscaled_gradients') else gradients

        embedding.assign_sub(delta) # Restore original embedding
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables)) # Update model weights

        # Update metrics and return results
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}

    def train_step(self, data):
        """Conditionally calls either the standard train_step or the FGM-perturbed train_step."""

        return tf.cond(self._train_counter < self.start_step,
                       lambda:super(FGM, self).train_step(data),
                       lambda:self.train_step_fgm(data))

class AWP(tf.keras.Model):
    """
    **Adversarial Weight Perturbation for Adversarial Training**

    This class implements Adversarial Weight Perturbation (AWP) as a Keras model for adversarial training.
    AWP perturbs all model weights, not just the embedding, for potentially stronger robustness.
    """

    def __init__(self,
                 *args,
                 delta=0.1,
                 eps=1e-4,
                 start_step=0,
                 **kwargs):
        
        super().__init__(*args, **kwargs)
        self.delta = delta              # Perturbation magnitude
        self.eps = eps                  # Small constant for numerical stability
        self.start_step = start_step    # Step at which to start applying AWP
        
    def train_step_awp(self, data):
        """
        Performs a single training step with AWP perturbation.
        """

        # source: https://stackoverflow.com/questions/62786227/how-to-track-weights-and-gradients-in-a-keras-custom-training-loop
        x, y = data # Unpack the input data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)                                 # Forward pass
            loss = self.compiled_loss(y, 
                                      y_pred, 
                                      regularization_losses=self.losses)    # Calculate loss

        # Calculate adversarial perturbations for all trainable variables
        params = self.trainable_variables                                   # Get all trainable variables
        params_gradients = tape.gradient(loss, self.trainable_variables)    # Calculate gradients

        # Apply AWP perturbations to each trainable variable
        for i in range(len(params_gradients)):
            grad = tf.zeros_like(params[i]) + params_gradients[i]           # Ensure non-zero gradients
            delta = tf.math.divide_no_nan(self.delta * grad , 
                                          tf.math.sqrt(tf.reduce_sum(grad**2)) + self.eps)
            self.trainable_variables[i].assign_add(delta)                   # Add perturbation to the variable

        with tf.GradientTape() as tape2:
            y_pred = self(x, training=True)                                 # Re-calculate predictions with perturbed weights
            new_loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
            if hasattr(self.optimizer, 'get_scaled_loss'):
                new_loss = self.optimizer.get_scaled_loss(new_loss)
        
        # Apply optimizer updates with potentially unscaled gradients
        gradients = tape2.gradient(new_loss, self.trainable_variables)
        gradients = self.optimizer.get_unscaled_gradients(gradients) if hasattr(self.optimizer, 'get_unscaled_gradients') else gradients

        for i in range(len(params_gradients)):
            grad = tf.zeros_like(params[i]) + params_gradients[i]           # Ensure non-zero gradients again
            delta = tf.math.divide_no_nan(self.delta * grad , tf.math.sqrt(tf.reduce_sum(grad**2)) + self.eps)
            params[i].assign_sub(delta)                                     # Remove perturbation before next step

        self.optimizer.apply_gradients(zip(gradients, 
                                           self.trainable_variables))       # Apply optimizer updates

        # Update metrics and return results
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}

    def train_step(self, data):
        """
        Conditionally calls standard or AWP training step based on start_step.
        """
        return tf.cond(self._train_counter < self.start_step,
                       lambda:super(AWP, self).train_step(data),
                       lambda:self.train_step_awp(data))