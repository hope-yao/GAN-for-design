from blocks.bricks.conv import (Convolutional, ConvolutionalSequence, ConvolutionalSequence, MaxPooling, Flattener )
from blocks.bricks import LeakyRectifier, Logistic, Softmax
from ali.utils import get_log_odds, conv_brick, conv_transpose_brick, bn_brick
from blocks.bricks.interfaces import Initializable, Random
from theano import tensor
from blocks.bricks.base import Brick, application, lazy

bs = 1
NLAT = 16
NEMB = 2
NUM_CHANNELS = 1
image_size = (1, 1)
RATIO = 16
LEAK = 1
dec_layers = [
    conv_transpose_brick(4, 1, 512 / RATIO), bn_brick(), LeakyRectifier(leak=LEAK),
    conv_transpose_brick(7, 2, 256 / RATIO), bn_brick(), LeakyRectifier(leak=LEAK),
    conv_transpose_brick(5, 2, 256 / RATIO), bn_brick(), LeakyRectifier(leak=LEAK),
    conv_transpose_brick(7, 2, 128 / RATIO), bn_brick(), LeakyRectifier(leak=LEAK),
    conv_transpose_brick(2, 1, 64 / RATIO), bn_brick(), LeakyRectifier(leak=LEAK),
    conv_brick(1, 1, NUM_CHANNELS), Logistic()]

class Decoder(Initializable):
    def __init__(self, layers, num_channels, image_size, use_bias=False, **kwargs):
        self.layers = layers
        self.num_channels = num_channels
        self.image_size = image_size

        self.mapping = ConvolutionalSequence(layers=layers,
                                             num_channels=num_channels,
                                             image_size=image_size,
                                             use_bias=use_bias,
                                             name='decoder_mapping')
        children = [self.mapping]
        kwargs.setdefault('children', []).extend(children)
        super(Decoder, self).__init__(**kwargs)

    @application(inputs=['z', 'y'], outputs=['outputs'])
    def apply(self, z, y, application_call):
        # Concatenating conditional data with inputs
        z_y = tensor.concatenate([z, y], axis=1)
        return self.mapping.apply(z_y)


class NewDecoder(Initializable):
    def __init__(self, layers, num_channels, image_size, use_bias=False, **kwargs):
        self.layers = layers
        self.num_channels = num_channels
        self.image_size = image_size

        self.mapping0 = ConvolutionalSequence(layers=layers[:-5],
                                             num_channels=num_channels-1,
                                             image_size=image_size,
                                             use_bias=use_bias,
                                             name='decoder_mapping0')
        self.mapping1 = ConvolutionalSequence(layers=layers[:-5],
                                             num_channels=num_channels-1,
                                             image_size=image_size,
                                             use_bias=use_bias,
                                             name='decoder_mapping1')
        self.mapping = ConvolutionalSequence(layers=layers[-5:],
                                             num_channels=8,
                                             image_size=(63,63),
                                             use_bias=use_bias,
                                             name='decoder_mapping')
        children = [self.mapping,self.mapping0,self.mapping1]
        kwargs.setdefault('children', []).extend(children)
        super(NewDecoder, self).__init__(**kwargs)

    @application(inputs=['z', 'y'], outputs=['outputs'])
    def apply(self, z, y, application_call):
        # network with branches and concatenation
        y0 = y[:,0:0,:,:]
        y1 = y[:,1:1,:,:]
        z_y0 = tensor.concatenate([z, y0], axis=1)
        z_y1 = tensor.concatenate([z, y1], axis=1)
        z_y = ( self.mapping0.apply(z_y0) + self.mapping1.apply(z_y1) )
        # z_y = tensor.concatenate([z, y], axis=1)
        z_yy = ( self.mapping.apply(z_y)  )
        return  z_yy


from blocks.initialization import IsotropicGaussian, Constant
decoder = NewDecoder(layers=dec_layers, num_channels=(NLAT + NEMB), image_size=(1, 1), use_bias=False,
    name='decoder_mapping0',  weights_init = IsotropicGaussian(0.01))
# decoder = Decoder(layers=dec_layers, num_channels=(NLAT + NEMB), image_size=(1, 1),
#                   weights_init=IsotropicGaussian(0.01))
decoder.initialize()

import numpy as np
z = np.zeros((bs,NLAT,1,1),dtype='float32')
y = np.zeros((bs,2,1,1),dtype='float32')
print ( decoder.apply(z,y).eval())
