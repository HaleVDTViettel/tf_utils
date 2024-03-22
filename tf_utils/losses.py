import math
import tensorflow as tf

def categorical_focal_loss(y_true,
                           y_pred,
                           gamma=2.0,
                           alpha=0.25,
                           label_smoothing=0,
                           from_logits=False,
                           sparse=False):
    """
    Focal Loss for Multi-Class Classification

    This function implements the Focal Loss as described in the paper "Focal Loss for Dense Object Detection" by Lin et al.
    Focal Loss addresses the class imbalance issue in multi-class classification by down-weighting the loss for well-classified 
    examples and focusing on improving the classification for hard-to-classify examples.

    Formula:
        loss = - gt * alpha * ((1 - pr)^gamma) * log(pr)
    
    Args:
        y_true: Ground truth labels, either in one-hot encoded form (sparse=False) or as integers (sparse=True).
        y_pred: Predicted logits or probabilities.
        gamma: Focusing parameter controlling the degree of down-weighting for well-classified examples (higher gamma leads to more down-weighting).
        alpha: Class balancing parameter, used to give higher weights to under-represented classes.
        label_smoothing: Value between 0 and 1 used for label smoothing (reducing overfitting).
        from_logits: Boolean indicating whether y_pred is logits (True) or probabilities (False).
        sparse: Boolean indicating whether y_true is one-hot encoded (False) or integers (True).

    Returns:
        The calculated focal loss.
    """

    y_pred = tf.cast(y_pred, tf.float32)    # Cast predictions to float32

    # One-hot encode labels if needed
    if sparse: 
        y_true = tf.one_hot(y_true,
                            tf.shape(y_pred)[-1],
                            axis=-1,
                            dtype=tf.float32) 
        
    if from_logits:
        y_pred = tf.nn.softmax(y_pred,
                               axis=-1)
    
    y_true = tf.cast(y_true, tf.float32)    # Cast ground truth labels to float32

    # Clip predictions to prevent NaNs and Infs during calculations
    y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1.0 - tf.keras.backend.epsilon())

    num_classes = tf.cast(tf.shape(y_true)[-1], tf.float32)                     # Get number of classes
    y_true = y_true * (1.0 - label_smoothing) + (label_smoothing / num_classes) # Apply label smoothing

    # Calculate focal loss based on the formula: loss = - gt * alpha * ((1 - pr)^gamma) * log(pr)
    loss = - y_true * (alpha * tf.pow((1 - y_pred), gamma) * tf.math.log(y_pred))
    return loss

def sv_softmax_loss(y_true,
                    logits,
                    t=1.2,
                    s=1,
                    label_smoothing=0,
                    normalize=False,
                    sparse=False):
    """
    SV-Softmax Loss for Multi-Class Classification

    This function implements the SV-Softmax loss as described in the paper "Learning Deep Networks with Label Smoothing" by Huang et al.
    SV-Softmax loss improves the generalization performance of deep neural networks by introducing a learnable temperature parameter (t).

    Args:
        y_true: Ground truth labels, either in one-hot encoded form (sparse=False) or as integers (sparse=True).
        logits: Network predictions before softmax.
        t: Temperature parameter controlling the softness of the softmax distribution (higher t leads to softer distribution).
        s: Scaling factor for the loss.
        label_smoothing: Value between 0 and 1 used for label smoothing (reducing overfitting).
        normalize: Boolean indicating whether to L2-normalize logits before calculation.
        sparse: Boolean indicating whether y_true is one-hot encoded (False) or integers (True).

    Returns:
        The calculated SV-Softmax loss.
    """

    #https://github.com/comratvlad/sv_softmax/blob/master/src/custom_losses.py

    logits = tf.cast(logits, tf.float32)            # Cast logits to float32
    epsilon = tf.keras.backend.epsilon()            # Small value to avoid division by zero
    zeros = tf.zeros_like(logits, dtype=tf.float32) # Tensor of zeros with same shape as logits
    ones = tf.ones_like(logits, dtype=tf.float32)   # Tensor of ones with

    # One-hot encode labels if needed
    if sparse:
        y_true = tf.one_hot(y_true,
                            tf.shape(logits)[-1],
                            axis=-1,
                            dtype=tf.float32)
        
    # L2-normalize logits if specified
    if normalize:
        logits = tf.math.l2_normalize(logits,
                                      axis=-1)
    
    y_true = tf.cast(y_true, tf.float32)            # Cast ground truth labels to float32

    logit_y = tf.reduce_sum(tf.multiply(y_true, logits),
                            axis=-1,
                            keepdims=True)          # Calculate weighted sum of logits and labels
    
    # Calculate intermediate terms for SV-Softmax loss formula
    I_k = tf.where(logit_y >= logits,               # Indicator function for selecting elements
                   zeros,
                   ones)  

    h = tf.exp(s * tf.multiply(t - 1., tf.multiply(logits + 1., I_k)))

    # Calculate softmax with temperature parameter
    softmax = tf.exp(s * logits) / (tf.reduce_sum(tf.multiply(tf.exp(s * logits), h), 
                                                  axis=-1, keepdims=True) + epsilon)
    softmax = tf.add(softmax, epsilon)  # Add epsilon because log(0) = nan

    num_classes = tf.cast(tf.shape(y_true)[-1],tf.float32)                      # Get number of classes
    y_true = y_true * (1.0 - label_smoothing) + (label_smoothing / num_classes) # Apply label smoothing

    # Calculate loss using label smoothing and softmax
    loss = tf.multiply(y_true, -tf.math.log(softmax))
    return loss

def arcface_loss(y_true,
                 logits,
                 s=30,
                 m=0.5,
                 easy_margin=False,
                 label_smoothing=0,
                 sparse=False):
    """
    ArcFace Loss for Multi-Class Classification with Margin

    This function implements the ArcFace loss as described in the paper "Additive Angular Margin Loss for Face Recognition" by Deng et al.
    ArcFace loss improves the discriminative power of deep face recognition models by introducing a margin between the class embeddings and an additive cosine margin.

    Args:
        y_true: Ground truth labels, either in one-hot encoded form (sparse=False) or as integers (sparse=True).
        logits: Network predictions before softmax.
        s: Scaling factor for the loss.
        m: Margin value for the ArcFace loss.
        easy_margin: Boolean indicating whether to use a simplified margin calculation.
        label_smoothing: Value between 0 and 1 used for label smoothing (reducing overfitting).
        sparse: Boolean indicating whether y_true is one-hot encoded (False) or integers (True).

    Returns:
        The calculated ArcFace loss.
    """

    logits = tf.cast(logits, tf.float32)    # Cast logits to float32

    # One-hot encode labels if needed
    if sparse:
        y_true = tf.one_hot(y_true,
                            tf.shape(logits)[-1],
                            axis=-1,
                            dtype=tf.float32)
        
    y_true = tf.cast(y_true, tf.float32)                                        # Cast ground truth labels to float32
    num_classes = tf.cast(tf.shape(y_true)[-1], tf.float32)                     # Get number of classes
    y_true = y_true * (1.0 - label_smoothing) + (label_smoothing / num_classes) # Apply label smoothing

    cosine = tf.math.l2_normalize(logits, axis=-1)      # L2-normalize logits for cosine similarity calculation
    sine = tf.math.sqrt(1.0 - tf.math.pow(cosine, 2))   # Calculate sine from cosine

    # Pre-calculate constants based on margin value (m)
    cos_m = tf.math.cos(m)
    sin_m = tf.math.sin(m)
    th = tf.math.cos(math.pi - m)
    mm = tf.math.sin(math.pi - m) * m

    # Calculate intermediate terms for ArcFace loss formula
    phi = cosine * cos_m - sine * sin_m
    if easy_margin:
        phi = tf.where(cosine > 0, phi, cosine)         # Simplified margin for easy margin
    else:
        phi = tf.where(cosine > th, phi, cosine - mm)   # Standard margin calculation

    # Calculate final output for ArcFace loss
    output = (y_true * phi) + ((1.0 - y_true) * cosine)                 # Combine predictions with margin and cosine similarity
    output *= s                                                         # Scale the output
    loss = tf.keras.losses.categorical_crossentropy(y_true,             # Calculate cross-entropy loss
                                                    output,
                                                    from_logits=True)   
    return loss

class CategoricalFocalCrossentropy(tf.keras.losses.Loss):
    """
    Keras Loss Class for Categorical Focal Loss

    This class wraps the `categorical_focal_loss` function into a Keras loss function for convenient use in model compilation.
    """

    def __init__(self,
                 gamma=2.0,
                 alpha=0.25,
                 label_smoothing=0,
                 from_logits=True,
                 sparse=True,
                 **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.alpha = alpha
        self.label_smoothing = label_smoothing
        self.from_logits = from_logits
        self.sparse = sparse

    def call(self, y_true, y_pred):
        """
        Calculates the Focal Loss given true labels and predictions.
        """

        loss = categorical_focal_loss(y_true, y_pred,
                                      gamma=self.gamma,
                                      alpha=self.alpha,
                                      label_smoothing=self.label_smoothing,
                                      from_logits=self.from_logits,
                                      sparse=self.sparse)
        N = tf.cast(tf.shape(y_true)[0], tf.float32) # Get number of samples

        return tf.reduce_sum(loss) / N # Calculate average loss per sample

class SVSoftmaxLoss(tf.keras.losses.Loss):
    """
    Keras Loss Class for SV-Softmax Loss

    This class implements the SV-Softmax Loss as a Keras loss function. It allows for easy integration with Keras models.
    """

    def __init__(self,
                 t=1.2,
                 s=1,
                 label_smoothing=0,
                 normalize=False,
                 sparse=True,
                 **kwargs):
        
        super().__init__(**kwargs)
        self.t = t
        self.s = s
        self.label_smoothing = label_smoothing
        self.normalize = normalize
        self.sparse = sparse

    def call(self, y_true, logits):
        """
        Calculates the SV-Softmax Loss given true labels and logits.
        """

        loss = sv_softmax_loss(y_true, logits,
                               s=self.s,
                               t=self.t,
                               label_smoothing=self.label_smoothing,
                               normalize=self.normalize,
                               sparse=self.sparse)
        N = tf.cast(tf.shape(y_true)[0], tf.float32) # Get number of samples

        return tf.reduce_sum(loss) / N # Calculate average loss per sample

class ArcfaceLoss(tf.keras.losses.Loss):
    """
    Keras Loss Class for ArcFace Loss

    This class implements the ArcFace Loss as a Keras loss function. It allows for easy integration with Keras models.
    """

    def __init__(self,
                 s=30,
                 m=0.3,
                 easy_margin=False,
                 label_smoothing=0,
                 sparse=True,
                 **kwargs):
        
        super().__init__(**kwargs)
        self.m = m
        self.s = s
        self.easy_margin = easy_margin
        self.label_smoothing = label_smoothing
        self.sparse = sparse

    def call(self, y_true, logits):
        """
        Calculates the ArcFace Loss given true labels and logits.
        """

        loss = arcface_loss(y_true, logits,
                               s=self.s,
                               m=self.m,
                               easy_margin=self.easy_margin,
                               label_smoothing=self.label_smoothing,
                               sparse=self.sparse)
        N = tf.cast(tf.shape(y_true)[0], tf.float32) # Get number of samples

        return tf.reduce_sum(loss) / N # Calculate average loss per sample