import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import concatenate

class UnetModel(tf.keras.Model):
    """
    This class implements the U-Net model.
    Parameters:
    - n_filters: The number of filters in the convolutional layers.
    - dropout_prob: The dropout probability.
    - max_pooling: Boolean, whether to use max pooling or not.
    - kernel_size: The size of the convolutional kernel.
    - input_size: The size of the input images.
    - n_classes: The number of classes in the segmentation task.
    """
    def __init__(self, n_filters, dropout_prob, max_pooling, kernel_size,
                 input_size=(128, 128, 3), n_classes=7):
        super(UnetModel, self).__init__()
        self.n_filters = n_filters
        self.dropout_prob = dropout_prob
        self.max_pooling = max_pooling
        self.kernel_size = kernel_size
        self.input_size = input_size
        self.n_classes = n_classes

    def _conv_block(self, n_filters, inputs=None, dropout_prob=0,
                    max_pooling=True):
        """
        Create a convolutional block.
        """
        conv = Conv2D(n_filters, self.kernel_size, activation='relu',
                      padding='same', kernel_initializer='he_normal')(inputs)
        conv = Conv2D(n_filters, self.kernel_size, activation='relu',
                      padding='same', kernel_initializer='he_normal')(conv)
        if dropout_prob > 0:
            conv = Dropout(dropout_prob)(conv)
        if max_pooling:
            next_layer = MaxPooling2D(pool_size=(2, 2))(conv)
        else:
            next_layer = conv
        skip_connection = conv
        return next_layer, skip_connection

    def _upsampling_block(self, n_filters, expansive_input, contractive_input):
        """
        Create an upsampling block.
        """
        up = Conv2DTranspose(n_filters, (self.kernel_size,
                             self.kernel_size), strides=(2,2),
                             padding='same')(expansive_input)
        merge = concatenate([up, contractive_input], axis=3)
        conv = Conv2D(n_filters, self.kernel_size, activation='relu',
                  padding='same', kernel_initializer='he_normal')(merge)
        conv = Conv2D(self.n_filters, self.kernel_size, activation='relu',
                  padding='same', kernel_initializer='he_normal')(conv)
        return conv

    def call(self):
        """
        Build the U-Net model.
        """
        inputs = Input(self.input_size)
        cblock1 = self._conv_block(self.n_filters, inputs)
        cblock2 = self._conv_block(2*self.n_filters, cblock1[0])
        cblock3 = self._conv_block(4*self.n_filters, cblock2[0])
        cblock4 = self._conv_block(8*self.n_filters, cblock3[0],
                                   self.dropout_prob)
        cblock5 = self._conv_block(16*self.n_filters, cblock4[0],
                                   self.dropout_prob,
                                   max_pooling=self.max_pooling)

        ublock6 = self._upsampling_block(8*self.n_filters, cblock5[0], cblock4[1])
        ublock7 = self._upsampling_block(4*self.n_filters, ublock6, cblock3[1])
        ublock8 = self._upsampling_block(2*self.n_filters, ublock7, cblock2[1])
        ublock9 = self._upsampling_block(self.n_filters, ublock8, cblock1[1])

        conv9 = Conv2D(self.n_filters, (self.kernel_size, self.kernel_size),
                       activation='relu', padding='same',
                       kernel_initializer='he_normal')(ublock9)

        conv10 = Conv2D(self.n_classes, (1,1), padding='same')(conv9)

        model = tf.keras.Model(inputs=inputs, outputs=conv10)
        return model

class DiceCCE(tf.keras.losses.Loss):
    def __init__(self, from_logits=False, smooth=1e-16, alpha=0.5, name="combined_dice_cce_loss"):
        """
        Combined loss function of Dice Loss and Categorical Cross-Entropy.
        Parameters:
        - from_logits: Boolean, whether `y_pred` is from logits (raw values) or probabilities.
        - smooth: Smoothing factor to prevent division by zero.
        - alpha: Weighting factor to control the contribution of Dice loss and Categorical Cross-Entropy loss.
        - name: Optional name for the loss function.
        """
        super().__init__(name=name)
        self.from_logits = from_logits
        self.smooth = smooth
        self.alpha = alpha
        self.cce = tf.keras.losses.CategoricalCrossentropy(from_logits=from_logits)

    def dice_loss(self, y_true, y_pred):
        """
        Computes the generalized Dice loss.
        Inputs:
        - y_true: Ground truth labels, expected to be one-hot encoded.
                  Shape: (batch_size, height, width, num_classes).
        - y_pred: Predicted labels, can be logits or probabilities.
                  Shape: (batch_size, height, width, num_classes).
        Returns:
        - loss: Computed Dice loss.
        """
        # If logits are passed, convert them to probabilities using softmax
        if self.from_logits:
            y_pred = tf.nn.softmax(y_pred)

        num_classes = tf.shape(y_true)[-1]
        y_true_f = tf.reshape(y_true, [-1, num_classes])
        y_pred_f = tf.reshape(y_pred, [-1, num_classes])

        y_true_f = tf.cast(y_true_f, tf.float32)
        y_pred_f = tf.cast(y_pred_f, tf.float32)

        intersection = tf.reduce_sum(y_true_f * y_pred_f, axis=0)
        union = tf.reduce_sum(y_true_f, axis=0) + tf.reduce_sum(y_pred_f, axis=0)

        dice_per_class = (2. * intersection + self.smooth) / (union + self.smooth)

        return 1.0 - dice_per_class

    def call(self, y_true, y_pred):
        """
        Computes the combined Dice loss and Categorical Cross-Entropy loss.

        Args:
        - y_true: Ground truth labels.
        - y_pred: Predicted labels.

        Returns:
        - loss: Combined loss (alpha * Dice loss + (1 - alpha) * Categorical Cross-Entropy loss).
        """
        # Calculate Dice loss
        dice_loss_value = self.dice_loss(y_true, y_pred)

        # Calculate Categorical Cross-Entropy loss
        cce_loss_value = self.cce(y_true, y_pred)

        # Combine the losses using the alpha weighting factor
        combined_loss = self.alpha * dice_loss_value + (1 - self.alpha) * cce_loss_value

        return combined_loss

class DiceLoss(tf.keras.losses.Loss):
    def __init__(self, from_logits=False, smooth=1e-16, name="dice_loss"):
        """
        Custom Dice Loss function for multi-class segmentation.
        Parameters:
        - from_logits: Boolean, whether `y_pred` is from logits (raw values) or probabilities.
        - smooth: Smoothing factor to prevent division by zero.
        - name: Optional name for the loss function.
        """
        super().__init__(name=name)
        self.from_logits = from_logits
        self.smooth = smooth

    def call(self, y_true, y_pred):
        """
        Computes the Dice loss.
        Inputs:
        - y_true: Ground truth labels, expected to be one-hot encoded.
                  Shape: (batch_size, height, width, num_classes).
        - y_pred: Predicted labels, can be logits or probabilities.
                  Shape: (batch_size, height, width, num_classes).
        Returns:
        - loss: Computed Dice loss.
        """
        # If logits are passed, convert them to probabilities using softmax
        if self.from_logits:
            y_pred = tf.nn.softmax(y_pred)

        # Calculate the number of classes
        num_classes = tf.shape(y_true)[-1]

        # Flatten the tensors to compute Dice coefficient
        y_true_f = tf.reshape(y_true, [-1, num_classes])
        y_pred_f = tf.reshape(y_pred, [-1, num_classes])

        # Cast to float32 for computation
        y_true_f = tf.cast(y_true_f, tf.float32)
        y_pred_f = tf.cast(y_pred_f, tf.float32)

        # Compute the intersection and the union for each class separately
        intersection = tf.reduce_sum(y_true_f * y_pred_f, axis=0)  # Sum over all voxels, for each class
        union = tf.reduce_sum(y_true_f, axis=0) + tf.reduce_sum(y_pred_f, axis=0)  # Sum over all voxels, for each class

        # Compute Dice coefficient for each class
        dice_per_class = (2. * intersection + self.smooth) / (union + self.smooth)

        # Return the Dice loss (1 - Dice coefficient)
        return 1.0 - dice_per_class