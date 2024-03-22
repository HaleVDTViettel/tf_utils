import tensorflow as tf
import matplotlib.pyplot as plt

class OneCycleLR(tf.keras.optimizers.schedules.LearningRateSchedule):

    """
    This class implements a unified single-cycle learning rate scheduler for TensorFlow.

    A single-cycle learning rate scheduler follows a specific pattern for adjusting the learning rate during training. 
    It typically consists of three phases:
        - Warmup: Gradually increase the learning rate from a minimum value to a peak value.
        - Sustain: Maintain the peak learning rate for a certain number of epochs.
        - Decay: Gradually decrease the learning rate from the peak value to a minimum value.

    This class allows you to configure various aspects of the learning rate schedule, including:
        - Initial learning rate (lr)
        - Total number of epochs (epochs)
        - Steps per epoch (steps_per_epoch)
        - Steps per update (steps_per_update) - How often the learning rate is updated within an epoch.
        - Resume epoch (resume_epoch) - Useful for resuming training from a checkpoint.
        - Decay epochs (decay_epochs) - Number of epochs for the decay phase.
        - Sustain epochs (sustain_epochs) - Number of epochs to maintain peak learning rate.
        - Warmup epochs (warmup_epochs) - Number of epochs for the warmup phase.
        - Minimum learning rate (lr_min)
        - Warmup type (warmup_type) - 'linear', 'exponential', or 'cosine' for warmup strategy.
        - Decay type (decay_type) - 'linear', 'exponential', or 'cosine' for decay strategy.

    """

    def __init__(self,
                lr=1e-4,
                epochs=10,
                steps_per_epoch=100,
                steps_per_update=1,
                resume_epoch=0,
                decay_epochs=10,
                sustain_epochs=0,
                warmup_epochs=0,
                lr_start=0,
                lr_min=0,
                warmup_type='linear',
                decay_type='cosine',
                **kwargs):
        """
        Initializes the OneCycleLR scheduler with the specified parameters.
        """
        
        super().__init__(**kwargs)
        self.lr = float(lr)                             # Learning rate
        self.epochs = float(epochs)                     # Total number of epochs
        self.steps_per_update = float(steps_per_update) # Steps per update
        self.resume_epoch = float(resume_epoch)         # Resume epoch
        self.steps_per_epoch = float(steps_per_epoch)   # Steps per epoch
        self.decay_epochs = float(decay_epochs)         # Decay epochs
        self.sustain_epochs = float(sustain_epochs)     # Sustain epochs
        self.warmup_epochs = float(warmup_epochs)       # Warmup epochs
        self.lr_start = float(lr_start)                 # Initial learning rate
        self.lr_min = float(lr_min)                     # Minimum learning rate
        self.decay_type = decay_type                    # Decay type
        self.warmup_type = warmup_type                  # Warmup type
        

    def __call__(self, step):
        """
        Calculates the learning rate for a given training step.

        Args:
            step: The current training step (integer or float).

        Returns:
            The calculated learning rate (float).
        """

        step = tf.cast(step, tf.float32)                            # Cast step to float32
        total_steps = self.epochs * self.steps_per_epoch            # Total training steps
        warmup_steps = self.warmup_epochs * self.steps_per_epoch    # Warmup steps
        sustain_steps = self.sustain_epochs * self.steps_per_epoch  # Sustain steps
        decay_steps = self.decay_epochs * self.steps_per_epoch      # Decay steps

        # Adjust step based on resume epoch
        if self.resume_epoch > 0:
            step = step + self.resume_epoch * self.steps_per_epoch

        # Ensure step doesn't exceed decay step
        step = tf.cond(step > decay_steps, lambda :decay_steps, lambda :step)

        # Scale step for updates per epoch
        step = tf.math.truediv(step, self.steps_per_update) * self.steps_per_update

        # Define conditions for warmup, decay, and sustain phases
        warmup_cond = step < warmup_steps
        decay_cond = step >= (warmup_steps + sustain_steps)
        
        # Implement different warmup strategies (linear, exponential, cosine)
        if self.warmup_type == 'linear':
            lr = tf.cond(warmup_cond,
                         lambda: tf.math.divide_no_nan(self.lr-self.lr_start, warmup_steps) * step + self.lr_start,
                         lambda: self.lr)
            
        elif self.warmup_type == 'exponential':
            factor = tf.pow(self.lr_start, 1/warmup_steps)
            lr = tf.cond(warmup_cond,
                         lambda: (self.lr - self.lr_start) * factor**(warmup_steps - step) + self.lr_start,
                         lambda: self.lr)

        elif self.warmup_type == 'cosine':
            lr = tf.cond(warmup_cond,
                         lambda: 0.5 * (self.lr - self.lr_start) * (1 + tf.cos(3.14159265359 * (warmup_steps - step)  / warmup_steps)) + self.lr_start,
                         lambda:self.lr)
        else:
            raise NotImplementedError(f"Unsupported warmup type: {self.warmup_type}")
                    
        # Implement different decay strategies (linear, exponential, cosine)
        if self.decay_type == 'linear':
            lr = tf.cond(decay_cond,
                         lambda: self.lr + (self.lr_min-self.lr)/(decay_steps - warmup_steps - sustain_steps)*(step - warmup_steps - sustain_steps),
                         lambda:lr)
            
        elif self.decay_type == 'exponential':
            factor = tf.pow(self.lr_min, 1/(decay_steps - warmup_steps - sustain_steps))
            lr = tf.cond(decay_cond,
                         lambda: (self.lr - self.lr_min) * factor**(step - warmup_steps - sustain_steps) + self.lr_min,
                         lambda:lr)
            
        elif self.decay_type == 'cosine':
            lr = tf.cond(decay_cond,
                         lambda: 0.5 * (self.lr - self.lr_min) * (1 + tf.cos(3.14159265359 * (step - warmup_steps - sustain_steps) / (decay_steps - warmup_steps - sustain_steps))) + self.lr_min,
                         lambda:lr)
        else:
            raise NotImplementedError(f"Unsupported decay type: {self.decay_type}")
            
        return lr

    def plot(self):
        """
        Plots the learning rate schedule.

        This function generates a scatter plot of the learning rate at each training step.
        """

        step = max(1, int(self.epochs*self.steps_per_epoch)//1000) #1 for total_steps < 1000, total_steps//1000 else
        eps = list(range(0,int(self.epochs*self.steps_per_epoch),step))
        learning_rates = [self(x) for x in eps]
        plt.scatter(eps,learning_rates,2)
        plt.show()
       

class ListedLR(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    This class implements a learning rate schedule that combines multiple learning rate schedules.

    The `ListedLR` class allows you to define a list of learning rate schedules (`self.schedules`) 
    and applies them sequentially during training. Each schedule is applied for a certain number of epochs 
    specified by its `epochs` attribute.

    """
    def __init__(self,
                 schedules,
                 steps_per_epoch=100,
                 update_per_epoch=1,
                 **kwargs):
        """
        Initializes the ListedLR scheduler with a list of learning rate schedules.

        Args:
            schedules: A list of learning rate schedule objects (e.g., OneCycleLR instances).
            steps_per_epoch: Number of training steps per epoch (default: 100).
            update_per_epoch: Number of times the learning rate is updated per epoch (default: 1).
        """

        super().__init__(**kwargs)
        self.schedules = schedules                      # List of learning rate schedules
        self.steps_per_epoch = float(steps_per_epoch)   # Steps per epoch
        self.update_per_epoch = float(update_per_epoch) # Updates per epoch

        # Update epochs and total steps for each schedule
        for s in self.schedules:
            s.steps_per_epoch = float(steps_per_epoch)
            s.update_per_epoch = float(update_per_epoch)
        
        # Calculate restart epochs (points where a new schedule starts)
        self.restart_epochs = tf.math.cumsum([s.epochs for s in self.schedules])
        self.epochs = self.restart_epochs[-1] # Total epochs for ListedLR

        # Calculate global steps (total steps for each schedule combined)
        self.global_steps = tf.math.cumsum([s.epochs * s.steps_per_epoch for s in self.schedules])
        
    def __call__(self, step):
        """
        Calculates the learning rate for a given training step.

        This function determines which learning rate schedule is active based on the current step 
        and then calls that schedule's `__call__` method to calculate the learning rate.

        Args:
            step: The current training step (integer or float).

        Returns:
            The calculated learning rate (float).
        """

        step = tf.cast(step, tf.float32)# Cast step to float32
        idx = tf.searchsorted(self.global_steps, [step+1])[0]   # Find the index of the active schedule using binary search
        global_steps = tf.concat([[0],self.global_steps],0)     # Concatenate global steps with a leading zero for indexing
        fns = [                                                 # Define a list of lambda functions, each calling the corresponding schedule's __call__ method
                (lambda x: (
                    lambda: self.schedules[x].__call__(step-global_steps[x])
                    )
                )(i) for i in range(len(self.schedules))
            ]
        r = tf.switch_case(idx, branch_fns=fns)                 # Use tf.switch_case to select the appropriate learning rate function based on the index
        return r

    def plot(self):
        """
        Plots the combined learning rate schedule.

        This function generates a scatter plot showing the learning rate for each training step 
        across all the listed learning rate schedules.
        """
        
        step = max(1, int(self.epochs*self.steps_per_epoch)//1000) #1 for total_steps < 1000, total_steps//1000 else
        eps = list(range(0,int(self.epochs*self.steps_per_epoch),step))
        learning_rates = [self(x) for x in eps]
        plt.scatter(eps,learning_rates,2)
        plt.show()