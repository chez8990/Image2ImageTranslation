"""
Implements a WGAN with U-net for realistic image generation
"""

from keras.models import Model
from keras.layers import *
from keras.constraints import Constraint
import keras.backend as K

class WeightClip(Constraint):
    '''
    Clips the weights incident to each hidden unit to be inside a range
    '''
    def __init__(self, c=2):
        self.c = c

    def __call__(self, p):
        return K.clip(p, -self.c, self.c)

    def get_config(self):
        return {'name': self.__class__.__name__,
                'c': self.c}

class RandomAverageInput(Layer):
    def build(self, input_shape):
        # self.in_shape = input_shape[0]

        self.built = True
    def call(self, inputs):
        x = inputs[0]
        z = inputs[1]

        self.shape = x.get_shape

        l = np.random.uniform(0, 1)

        return (1 - l) * x + l * z
    def compute_output_shape(self, input_shape):
        return input_shape[0]

def W_loss(y_true, y_pred):
    """
    See https://arxiv.org/pdf/1701.07875.pdf for more details
    input of y_true is a binary variable, 0 means fake 1 means real
    :param y_true:
    :param y_pred:
    :return:
    """

    sign = 1 - y_true * 2

    return K.mean(sign * y_pred)


def generator(ngf=16, x_shape=(64, 64, 1)):
    # crate encoder for the generator
    kernel_size = 4
    strides = 2
    padding = 'same'

    layer_specs = [
                   ngf * 2,  # encoder_2: [batch, 128, 128,ngf] => [batch, 64, 64,ngf * 2]
                   ngf * 4,  # encoder_3: [batch, 64, 64,ngf * 2] => [batch, 32, 32,ngf * 4]
                   ngf * 8,  # encoder_4: [batch, 32, 32,ngf * 4] => [batch, 16, 16,ngf * 8]
                   ngf * 8,  # encoder_5: [batch, 16, 16,ngf * 8] => [batch, 8, 8,ngf * 8]
                   ngf * 8,  # encoder_6: [batch, 8, 8,ngf * 8] => [batch, 4, 4,ngf * 8]
                   ngf * 8,  # encoder_7: [batch, 4, 4,ngf * 8] => [batch, 2, 2,ngf * 8]
                   # ngf * 8,  # encoder_8: [batch, 2, 2,ngf * 8] => [batch, 1, 1,ngf * 8]
                   ]

    layer_outputs = [] # store outputs for skip connections

    # initialize inputs
    x_input = Input(x_shape)
    z_input = Input(shape=(ngf * 8, ))

    x = Conv2D(ngf, kernel_size=kernel_size, strides=strides, padding=padding)(x_input)
    # x = BatchNormalization()(x)
    x = Activation('relu', name='encoder_layer0')(x)

    layer_outputs.append(x)

    for i, output_channels in enumerate(layer_specs):
        x = Conv2D(output_channels, kernel_size=kernel_size, strides=strides, padding=padding)(x)
        x = BatchNormalization()(x)
        x = Activation('relu', name='encoder_layer{}'.format(i+1))(x)

        if i < len(layer_specs) - 1:
            layer_outputs.append(x)

    # add noise in the latent space
    z = Reshape((1, 1, ngf * 8))(z_input)

    x = Concatenate(axis=-1)([x, z])

    # decoder network
    layer_specs = [#ngf * 8,
                   ngf * 8,  # decoder_7: [batch, 2, 2, ngf * 8 * 2] => [batch, 4, 4, ngf * 8 * 2]
                   ngf * 8,  # decoder_6: [batch, 4, 4, ngf * 8 * 2] => [batch, 8, 8, ngf * 8 * 2]
                   ngf * 8,  # decoder_5: [batch, 8, 8, ngf * 8 * 2] => [batch, 16, 16, ngf * 8 * 2]
                   ngf * 4,  # decoder_4: [batch, 16, 16, ngf * 8 * 2] => [batch, 32, 32, ngf * 4 * 2]
                   ngf * 2,  # decoder_3: [batch, 32, 32, ngf * 4 * 2] => [batch, 64, 64, ngf * 2 * 2]
                   ngf,       # decoder_2: [batch, 64, 64, ngf * 2 * 2] => [batch, 128, 128, ngf * 2]
                   3]

    for i, output_channels in enumerate(layer_specs):
        if i > 0:
            x = Concatenate(axis=-1)([layer_outputs[-1], x])
            layer_outputs.pop(-1)

        x = UpSampling2D((2, 2))(x)
        x = Conv2D(output_channels, kernel_size=kernel_size, padding=padding)(x)
        x = BatchNormalization()(x)

        if i == len(layer_specs) - 1:
            print(i, x)
            x = Activation('tanh', name='decoder_layer{}'.format(i + 1))(x)
        else:
            x = Activation('relu', name='decoder_layer{}'.format(i + 1))(x)


    model = Model(inputs=[x_input, z_input], outputs=x)

    return model

def discriminator(ngf=16, image_shape=(64, 64, 3), weight_clipping=False, last_activation='sigmoid'):
    kernel_size = 4
    strides = 2

    x_input = Input(shape=image_shape)
    z_input = Input(shape=image_shape)

    # x = Concatenate(axis=-1)([x_input, z_input])

    x = RandomAverageInput()([x_input, z_input])

    if weight_clipping == True:
        for i in range(4):
            x = Conv2D(ngf * (i+1)//2 ,
                       kernel_size=kernel_size,
                       strides=strides,
                       kernel_constraint=WeightClip(.1))(x)
            # x = BatchNormalization()(x)
            x = Activation('relu')(x)


        x = Flatten()(x)
        x = Dense(1024, kernel_initializer='he_normal', kernel_constraint=WeightClip(.01))(x)
        x = LeakyReLU(.2)(x)
        x = Dropout(.3)(x)
        x = Dense(1, kernel_initializer='he_normal', kernel_constraint=WeightClip(.01), activation=last_activation)(x)

    else:
        for i in range(4):
            x = Conv2D(ngf * (i+1)//2 ,
                       kernel_size=kernel_size,
                       strides=strides)(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)


        x = Flatten()(x)
        x = Dense(1024, kernel_initializer='he_normal')(x)
        x = LeakyReLU(.2)(x)
        x = Dropout(.3)(x)
        x = Dense(1, kernel_initializer='he_normal', activation=last_activation)(x)

    model = Model(inputs=[x_input, z_input], outputs=x)

    return model
