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
# dec_layers = [
#     conv_transpose_brick(4, 1, 512 / RATIO), bn_brick(), LeakyRectifier(leak=LEAK),
#     conv_transpose_brick(7, 2, 256 / RATIO), bn_brick(), LeakyRectifier(leak=LEAK),
#     conv_transpose_brick(5, 2, 256 / RATIO), bn_brick(), LeakyRectifier(leak=LEAK),
#     conv_transpose_brick(7, 2, 128 / RATIO), bn_brick(), LeakyRectifier(leak=LEAK),
#     conv_transpose_brick(2, 1, 64 / RATIO), bn_brick(), LeakyRectifier(leak=LEAK),
#     conv_brick(1, 1, NUM_CHANNELS), Logistic()]

dec_layers_sub0 = [
    conv_transpose_brick(4, 1, 512 / RATIO), bn_brick(), LeakyRectifier(leak=LEAK),
    conv_transpose_brick(7, 2, 256 / RATIO), bn_brick(), LeakyRectifier(leak=LEAK),
    conv_transpose_brick(5, 2, 256 / RATIO), bn_brick(), LeakyRectifier(leak=LEAK),
    conv_transpose_brick(7, 2, 128 / RATIO), bn_brick(), LeakyRectifier(leak=LEAK)]
dec_layers_sub1 = [
    conv_transpose_brick(4, 1, 512 / RATIO), bn_brick(), LeakyRectifier(leak=LEAK),
    conv_transpose_brick(7, 2, 256 / RATIO), bn_brick(), LeakyRectifier(leak=LEAK),
    conv_transpose_brick(5, 2, 256 / RATIO), bn_brick(), LeakyRectifier(leak=LEAK),
    conv_transpose_brick(7, 2, 128 / RATIO), bn_brick(), LeakyRectifier(leak=LEAK)]
dec_layers_sum = [
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
    def __init__(self, dec_layers_sub0, dec_layers_sub1, dec_layers_sum, num_channels, image_size, use_bias=False, **kwargs):
        # self.layers = layers
        self.num_channels = num_channels
        self.image_size = image_size

        self.mapping0 = ConvolutionalSequence(layers=dec_layers_sub0,
                                             num_channels=num_channels-1,
                                             image_size=image_size,
                                             use_bias=use_bias,
                                             name='decoder_mapping0')
        self.mapping1 = ConvolutionalSequence(layers=dec_layers_sub1,
                                             num_channels=num_channels-1,
                                             image_size=image_size,
                                             use_bias=use_bias,
                                             name='decoder_mapping1')
        self.mapping = ConvolutionalSequence(layers=dec_layers_sum,
                                             num_channels=8,
                                             image_size=(63,63),
                                             use_bias=use_bias,
                                             name='decoder_mapping')
        children = [self.mapping0,self.mapping,self.mapping1]
        kwargs.setdefault('children', []).extend(children)
        super(NewDecoder, self).__init__(**kwargs)

    @application(inputs=['z', 'y'], outputs=['outputs'])
    def apply(self, z, y, application_call):
        y0, y1 = y
        z_y0 = tensor.concatenate([z, y0], axis=1)
        z_y1 = tensor.concatenate([z, y1], axis=1)
        return  self.mapping.apply(self.mapping0.apply(z_y0)+ self.mapping1.apply(z_y1))


from blocks.initialization import IsotropicGaussian, Constant
decoder = NewDecoder(dec_layers_sub0, dec_layers_sub1, dec_layers_sum, num_channels=(NLAT + NEMB), image_size=(1, 1), use_bias=False,
    name='decoder_mapping0',  weights_init = IsotropicGaussian(0.01))
# decoder = Decoder(layers=dec_layers, num_channels=(NLAT + NEMB), image_size=(1, 1),
#                   weights_init=IsotropicGaussian(0.01))
decoder.initialize()

# import numpy as np
# z = np.zeros((bs,NLAT,1,1),dtype='float32')
# # y0 = np.zeros((bs,1,1,1),dtype='float32')
# # y1 = np.zeros((bs,1,1,1),dtype='float32')
# y = np.zeros((bs,2,1,1),dtype='float32')
# y0, y1 = y[:,0:1,:,:], y[:,1:,:,:]
# print ( decoder.apply(z,[y0,y1]).eval())





import numpy as np
data = np.zeros((128,18,1,1))
label = np.zeros((128,1,64,64))
split = 0.2
l = len(data) #length of data
n1 = int(split*l)  # split for testing
n2 = l - n1
from random import sample
indices = sample(range(l),n1)

data_test = data[indices]
label_test = label[indices]

data_train = np.delete(data,indices,0)
label_train = np.delete(label,indices,0)
import h5py

train_features = []
train_targets = []
test_features = []
test_targets = []
for index, array in enumerate(data_train):
    train_features.append(array.reshape(array.shape[0],array.shape[1],array.shape[2]))
    train_targets.append(label_train[index])
for index, array in enumerate(data_test):
    test_features.append(array.reshape(array.shape[0],array.shape[1],array.shape[2]))
    test_targets.append(label_test[index])

train_features = np.array(train_features)
train_targets = np.array(train_targets) #starts from 0
test_features = np.array(test_features)
test_targets = np.array(test_targets)
train_n, c, p1, p2 = train_features.shape
test_n = test_features.shape[0]
n = train_n + test_n

f = h5py.File('/home/hope-yao/Documents/Data/test.hdf5', mode='w')
features = f.create_dataset('features', (n, c, p1, p2), dtype='uint8')
m = 64
targets = f.create_dataset('targets', (n, m, m), dtype='uint8')

features[...] = np.vstack([train_features, test_features])
targets[...] = np.vstack([train_targets, test_targets]).reshape(n,m,m)

features.dims[0].label = 'batch'
features.dims[1].label = 'channel'
features.dims[2].label = 'height'
features.dims[3].label = 'width'
targets.dims[0].label = 'batch'
targets.dims[1].label = 'targets'


from fuel.datasets.hdf5 import H5PYDataset
split_dict = {
    'train': {'features': (0, train_n), 'targets': (0, train_n)},
    'valid': {'features': (train_n, n), 'targets': (train_n, n)}}
f.attrs['split'] = H5PYDataset.create_split_array(split_dict)

f.flush()
f.close()






from blocks.algorithms import GradientDescent, CompositeRule, Restrict
from collections import OrderedDict
from blocks.model import Model
from theano import tensor, grad
from blocks.select import Selector
from blocks.main_loop import MainLoop
from blocks.algorithms import Adam, RMSProp, Momentum

zy = tensor.tensor4('features')
x = tensor.matrix('targets')
z,y0,y1 = zy[:,0:16,:,:],zy[:,16:17,:,:],zy[:,17:,:,:]
pred = decoder.apply(z,[y0,y1])
classifier_cost = - (tensor.mean(x * tensor.log(pred)) + tensor.mean((1 - x) * tensor.log(1 - pred)))
gradients = OrderedDict()

classifier_parameters = list(Selector([decoder]).get_parameters().values())
gradients.update(
    zip(classifier_parameters,
        grad(classifier_cost, classifier_parameters)))
LEARNING_RATE_C = 1e-3
step_rule_c = RMSProp(learning_rate=LEARNING_RATE_C)
classify_algorithm = GradientDescent(cost=classifier_cost,
                                     gradients=gradients,
                                     parameters=classifier_parameters,
                                     step_rule=step_rule_c)
from ali.streams import create_celeba_data_streams, create_crs_data_streams, create_mnist64_data_streams
streams = create_crs_data_streams(16, 16,
                                  sources=('features', 'targets'))
main_loop_stream, train_monitor_stream, valid_monitor_stream = streams
from blocks.extensions import FinishAfter, Timing, Printing, ProgressBar
extensions = [
    Timing(),
    FinishAfter(after_n_epochs=4),
    ProgressBar(),
    Printing(),
]
classify_loop = MainLoop(data_stream=main_loop_stream,
                         model=Model(classifier_cost),
                         algorithm=classify_algorithm,
                         # extensions=extensions
                         )
print('classifier training...')
classify_loop.run()