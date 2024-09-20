import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import concatenate

class UnetModel(tf.keras.Model):
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
        conv = Conv2D(n_filters, self.kernel_size, activation='relu',
                      padding='same', kernel_initializer='he_normal')(inputs)
        conv = Conv2D(n_filters, self.kernel_size, activation='relu',
                      padding='same', kernel_initializer='he_normal')(conv)
        if dropout_prob > 0:
            conv = tf.keras.layers.Dropout(dropout_prob)(conv)
        if max_pooling:
            next_layer = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv)
        else:
            next_layer = conv
        skip_connection = conv
        return next_layer, skip_connection

    def _upsampling_block(self, n_filters, expansive_input, contractive_input):
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

        conv10 = Conv2D(self.n_classes, (1, 1), padding='same')(conv9)

        model = tf.keras.Model(inputs=inputs, outputs=conv10)
        return model
