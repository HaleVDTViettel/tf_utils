import tensorflow as tf

class Snapshot(tf.keras.callbacks.Callback):
    """
    **Checkpoint Manager**

    This callback helps us periodically save snapshots of the model's weights during training. 
    These snapshots can be used for:

    * Resuming training from a specific point if interrupted.
    * Evaluating the model's performance at different stages of training.
    * Experimenting with hyperparameter tuning using different weights.

    **Arguments:**

    * `save_name` (str): The base filename for the snapshots (e.g., 'mymodel').
    * `snapshot_epochs` (list, optional): A list of specific epochs where to save weights. Defaults to saving only the last epoch.
    """
    def __init__(self,save_name,snapshot_epochs=[]):
        super().__init__()
        self.snapshot_epochs = snapshot_epochs  # Remember the epochs for snapshots 
        self.save_name = save_name              # Base filename for snapshots  
        
    def on_epoch_end(self, epoch, logs=None):
        """
        Called at the end of each epoch.

        This function checks if the current epoch is included in the `snapshot_epochs` list.
        If so, it saves the model's weights with two filenames:

        * `{save_name}-epoch{epoch}.h5`: This file includes the epoch number in the filename for easy identification.
        * `{save_name}-last.h5`: This file is always overwritten with the latest weights, regardless of epoch.
        """
        if epoch in self.snapshot_epochs:
            print(f" Saving snapshot for epoch {epoch}")
            self.model.save_weights(f"{self.save_name}-epoch{epoch}.h5")    # Save with epoch number
        self.model.save_weights(f"{self.save_name}-last.h5")                # Always save the last weights
        
class SWA(tf.keras.callbacks.Callback): 
    """
    **Stochastic Weight Averaging**

    This callback implements Stochastic Weight Averaging (SWA) during training. SWA improves 
    generalization performance by averaging weights across multiple epochs.

    **Arguments:**

    * `save_name` (str): The base filename for the SWA weights file.
    * `swa_epochs` (list, optional): A list of epochs to include in the SWA averaging. Defaults to an empty list (no SWA).
    * `strategy` (tf.distribute.Strategy, optional): The distribution strategy used for training (if applicable).
    * `train_ds` (tf.data.Dataset, optional): The training dataset used for model training.
    * `valid_ds` (tf.data.Dataset, optional): The validation dataset used for model evaluation.
    * `train_steps` (int, optional): The number of steps per epoch in the training dataset. Defaults to 1000.
    * `valid_steps` (int, optional): The number of steps per epoch in the validation dataset (if applicable).
    """

    def __init__(self,save_name,swa_epochs=[],strategy=None,train_ds=None,valid_ds=None,train_steps=1000,valid_steps=None):
        super().__init__()
        self.swa_epochs = swa_epochs
        self.swa_weights = None
        self.save_name = save_name
        self.train_ds = train_ds
        self.valid_ds = valid_ds
        self.train_steps = train_steps
        self.valid_steps = valid_steps
        self.strategy = strategy
    
    @tf.function
    def train_step(self, iterator):
        """
        The step function for one training step within the SWA update process.
        (Internal function, not intended for separate use)
        """
        def step_fn(inputs):
            """
            The computation to run on each device during distributed training.
            (Internal function, not intended for separate use)
            """
            x,y = inputs
            _ = self.model(x, training=True)

        for x in iterator:
            self.strategy.run(step_fn, args=(x,))
            
    def on_epoch_end(self, epoch, logs=None):
        """
        Called at the end of each epoch.

        This function checks if the current epoch is included in the `swa_epochs` list.
        If so, it accumulates the model's weights for averaging later.
        """
        if epoch in self.swa_epochs:   
            if self.swa_weights is None:
                self.swa_weights = self.model.get_weights()
            else:
                w = self.model.get_weights()
                for i in range(len(self.swa_weights)):
                    self.swa_weights[i] += w[i] # Accumulate weights for averaging              
    
    def on_train_end(self, logs=None):
        """
        Called at the end of training.

        This function performs the final averaging of the accumulated weights and sets them as the model's weights.
        Additionally, it offers options for recalculating running statistics and evaluating the SWA model.
        """
        if len(self.swa_epochs):
            print('applying SWA...')
            for i in range(len(self.swa_weights)):
                self.swa_weights[i] = self.swa_weights[i]/len(self.swa_epochs)
            self.model.set_weights(self.swa_weights)

            if self.train_ds is not None:
                # Recalculate running mean and variance for better evaluation (optional)
                print('Recalculating running statistics...')
                self.train_step(self.train_ds.take(self.train_steps))

            print(f'save SWA weights to {self.save_name}-SWA.h5')
            self.model.save_weights(f"{self.save_name}-SWA.h5")

            if self.valid_ds is not None:
                # Evaluate the SWA model on the validation dataset (optional)
                print('Evaluating SWA model...')
                self.model.evaluate(self.valid_ds, steps=self.valid_steps)