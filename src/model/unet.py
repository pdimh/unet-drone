import numpy as np
import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras.backend import categorical_crossentropy, constant, mean
from tensorflow.keras.layers import BatchNormalization, concatenate, Conv2D, Conv2DTranspose, Dropout, Input,  MaxPooling2D


def conv2d(input, n_filters):
    conv1 = Conv2D(n_filters, (3, 3), activation='relu',
                   padding='same')(input)
    conv1 = BatchNormalization()(conv1)

    conv2 = Conv2D(n_filters, (3, 3), activation='relu',
                   padding='same')(conv1)
    conv2 = BatchNormalization()(conv2)
    return conv2


def get_model(input_size=(480, 720, 3), n_filters=16, dropout=0.1, n_classes=3, class_weights=None):

    if class_weights is None:
        class_weights = np.tile(1/n_classes, (1, n_classes))

    # @tf.autograph.experimental.do_not_convert
    def loss(y_true, y_pred):
        class_loglosses = mean(categorical_crossentropy(
            y_true, y_pred), axis=[0, 1, 2])
        return tf.math.reduce_sum(class_loglosses * constant(class_weights))

    inputs = Input(input_size)

    # Encode
    conv1 = conv2d(inputs, n_filters)
    pool1 = MaxPooling2D((2, 2))(conv1)
    drop1 = Dropout(dropout)(pool1)

    n_filters *= 2
    conv2 = conv2d(drop1, n_filters)
    pool2 = MaxPooling2D((2, 2))(conv2)
    drop2 = Dropout(dropout)(pool2)

    n_filters *= 2
    conv3 = conv2d(drop2, n_filters)
    pool3 = MaxPooling2D((2, 2))(conv3)
    drop3 = Dropout(dropout)(pool3)

    n_filters *= 2
    conv4 = conv2d(drop3, n_filters)
    pool4 = MaxPooling2D((2, 2))(conv4)
    drop4 = Dropout(dropout)(pool4)

    n_filters *= 2
    conv5 = conv2d(drop4, n_filters)

    # Decode
    n_filters /= 2
    transp6 = Conv2DTranspose(
        n_filters, (3, 3), strides=(2, 2), padding='same')(conv5)
    up6 = concatenate([transp6, conv4])
    conv6 = conv2d(up6, n_filters)
    drop6 = Dropout(dropout)(conv6)

    n_filters /= 2
    transp7 = Conv2DTranspose(
        n_filters * 4, (3, 3), strides=(2, 2), padding='same')(drop6)
    up7 = concatenate([transp7, conv3])
    conv7 = conv2d(up7, n_filters)
    drop7 = Dropout(dropout)(conv7)

    n_filters /= 2
    transp8 = Conv2DTranspose(
        n_filters * 2, (3, 3), strides=(2, 2), padding='same')(drop7)
    up8 = concatenate([transp8, conv2])
    conv8 = conv2d(up8, n_filters)
    drop8 = Dropout(dropout)(conv8)

    n_filters /= 2
    transp9 = Conv2DTranspose(
        n_filters * 1, (3, 3), strides=(2, 2), padding='same')(drop8)
    up9 = concatenate([transp9, conv1])
    conv9 = conv2d(up9, n_filters)
    drop9 = Dropout(dropout)(conv9)

    outputs = Conv2D(n_classes, (1, 1), activation='sigmoid')(drop9)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss=loss)

    return model
