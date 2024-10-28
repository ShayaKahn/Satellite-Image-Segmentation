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
    def __init__(self, from_logits=False, smooth=1e-16, use_class_weights=False, name="dice_loss"):
        """
        Custom Dice Loss function for multi-class segmentation with optional class weights.
        Parameters:
        - from_logits: Boolean, whether `y_pred` is from logits (raw values) or probabilities.
        - smooth: Smoothing factor to prevent division by zero.
        - use_class_weights: Boolean, whether to apply class weights based on `y_true`.
        - name: Optional name for the loss function.
        """
        super().__init__(name=name)
        self.from_logits = from_logits
        self.smooth = smooth
        self.use_class_weights = use_class_weights

    def compute_class_weights(self, y_true):
        """
        Computes class weights based on the number of voxels for each class.
        Inputs:
        - y_true: Ground truth labels, expected to be one-hot encoded.
                  Shape: (batch_size, height, width, num_classes).
        Returns:
        - class_weights: A tensor of shape (num_classes,) representing class weights.
        """
        # Calculate the number of classes
        num_classes = tf.shape(y_true)[-1]

        # Calculate the number of voxels for each class (R_l)
        voxel_counts_per_class = tf.reduce_sum(y_true, axis=[0, 1, 2])  # Sum over spatial dimensions

        # Total number of voxels in the dataset (N)
        total_voxels = tf.reduce_sum(voxel_counts_per_class)

        # Calculate class weights: W = N / (L * |R_l| + 1)
        class_weights = total_voxels / (tf.cast(num_classes, tf.float32) * voxel_counts_per_class + 1.0)

        # Normalize the weights to sum to 1
        class_weights /= tf.reduce_sum(class_weights)

        return class_weights

    def call(self, y_true, y_pred):
        """
        Computes the Dice loss with optional class weights.
        Inputs:
        - y_true: Ground truth labels, expected to be one-hot encoded.
                  Shape: (batch_size, height, width, num_classes).
        - y_pred: Predicted labels, can be logits or probabilities.
                  Shape: (batch_size, height, width, num_classes).
        Returns:
        - loss: Computed Dice loss with optional class weights.
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
        dice_loss_per_class = 1.0 - dice_per_class  # Dice loss per class

        # Optionally apply class weights if use_class_weights is True
        if self.use_class_weights:
            # Compute class weights based on the ground truth labels
            class_weights = self.compute_class_weights(y_true)
            # Weighted Dice loss: multiply each class loss by its weight
            weighted_dice_loss = dice_loss_per_class * class_weights
            return tf.reduce_mean(weighted_dice_loss)
        else:
            # Return the mean Dice loss if not using class weights
            return tf.reduce_mean(dice_loss_per_class)

class JaccardLoss(tf.keras.losses.Loss):
    def __init__(self, from_logits=False, smooth=1e-16, use_class_weights=False,
                 name="jaccard_loss"):
        """
        Custom Jaccard loss class for multi-class segmentation with optional class weights.
        Parameters:
        - from_logits: Boolean, whether `y_pred` is from logits (raw values) or probabilities.
        - smooth: Smoothing factor to prevent division by zero and ensure numerical stability.
        - use_class_weights: Boolean, whether to apply class weights based on `y_true`.
        - name: Optional name for the loss function.
        """
        super().__init__(name=name)
        self.from_logits = from_logits
        self.smooth = smooth
        self.use_class_weights = use_class_weights

    def compute_class_weights(self, y_true):
        """
        Computes class weights based on the number of voxels for each class.
        Inputs:
        - y_true: Ground truth labels, expected to be one-hot encoded.
                  Shape: (batch_size, height, width, num_classes).
        Returns:
        - class_weights: A tensor of shape (num_classes,) representing class weights.
        """
        # Calculate the number of classes
        num_classes = tf.shape(y_true)[-1]

        # Calculate the number of voxels for each class (R_l)
        voxel_counts_per_class = tf.reduce_sum(y_true, axis=[0, 1, 2])  # Sum over spatial dimensions

        # Total number of voxels in the dataset (N)
        total_voxels = tf.reduce_sum(voxel_counts_per_class)

        # Calculate class weights: W = N / (L * |R_l| + 1)
        class_weights = total_voxels / (tf.cast(num_classes, tf.float32) * voxel_counts_per_class + 1.0)

        # Normalize the weights to sum to 1 (optional, for numerical stability)
        class_weights /= tf.reduce_sum(class_weights)

        return class_weights

    def call(self, y_true, y_pred):
        """
        Computes the custom Jaccard loss with optional class weights.
        Inputs:
        - y_true: Ground truth labels, expected to be one-hot encoded.
                  Shape: (batch_size, height, width, num_classes).
        - y_pred: Predicted labels, can be logits or probabilities.
                  Shape: (batch_size, height, width, num_classes).
        Returns:
        - loss: Computed Jaccard loss with optional class weights.
        """
        # If logits are passed, convert them to probabilities using softmax
        if self.from_logits:
            y_pred = tf.nn.softmax(y_pred)

        # Flatten the tensors for calculation
        y_true_f = tf.reshape(y_true, [-1])
        y_pred_f = tf.reshape(y_pred, [-1])

        # Cast to float32 for computation
        y_true_f = tf.cast(y_true_f, tf.float32)
        y_pred_f = tf.cast(y_pred_f, tf.float32)

        # Compute the intersection and union
        intersection = tf.reduce_sum(y_true_f * y_pred_f)
        union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) - intersection

        # Compute Jaccard index and Jaccard loss
        jaccard_index = (intersection + self.smooth) / (union + self.smooth)
        jaccard_loss = 1.0 - jaccard_index

        # Optionally apply class weights if use_class_weights is True
        if self.use_class_weights:
            # Compute class weights based on the ground truth labels
            class_weights = self.compute_class_weights(y_true)
            # Multiply each class in y_true by its corresponding class weight
            weights = tf.reduce_sum(class_weights * y_true, axis=-1)  # Sum weights for each class in each sample
            weighted_loss = jaccard_loss * weights
            return tf.reduce_mean(weighted_loss)
        else:
            # If not using class weights, return the standard Jaccard loss
            return tf.reduce_mean(jaccard_loss)

class CombinedLoss(tf.keras.losses.Loss):
    def __init__(self, loss1, loss2, alpha=0.5, name="combined_loss"):
        """
        Custom combined loss function for combining two different loss functions.
        Parameters:
        - loss1: First loss function (instance of tf.keras.losses.Loss).
        - loss2: Second loss function (instance of tf.keras.losses.Loss).
        - alpha: Weighting factor for the first loss (0 <= alpha <= 1).
        - name: Optional name for the loss function.
        """
        super().__init__(name=name)
        self.loss1 = loss1
        self.loss2 = loss2
        self.alpha = alpha

    def call(self, y_true, y_pred):
        """
        Computes the combined loss as a weighted sum of the two losses using alpha.
        Inputs:
        - y_true: Ground truth labels.
        - y_pred: Predicted labels.
        Returns:
        - Combined loss value.
        """
        # Calculate each loss
        loss1_value = self.loss1(y_true, y_pred)
        loss2_value = self.loss2(y_true, y_pred)

        # Combine the losses with alpha and (1 - alpha)
        combined_loss = self.alpha * loss1_value + (1 - self.alpha) * loss2_value

        return combined_loss

class MeanIoU(tf.keras.metrics.MeanIoU):
    """
    This class extends the MeanIoU class from tf.keras.metrics.
    """
    def __init__(self, num_classes, name=None, dtype=None):
        super(MeanIoU, self).__init__(num_classes=num_classes,
                                      name=name, dtype=dtype)

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Convert logits to predicted class labels
        y_pred = tf.argmax(y_pred, axis=-1)
        y_true = tf.argmax(y_true, axis=-1)

        return super().update_state(y_true, y_pred, sample_weight)