""" Conditional ALI related Bricks"""

from theano import tensor
from theano import (function, )

from blocks.bricks.base import Brick, application, lazy
from blocks.bricks import LeakyRectifier, Logistic, Softmax
from blocks.bricks import (MLP, Rectifier, FeedforwardSequence, Linear, Sequence)
from blocks.bricks.conv import (Convolutional, ConvolutionalSequence, ConvolutionalSequence, MaxPooling, Flattener )
from blocks.bricks.interfaces import Initializable, Random
from toolz.itertoolz import interleave
import logging

from blocks.initialization import IsotropicGaussian, Constant

from blocks.select import Selector

from ali.bricks import ConvMaxout
from ali.utils import get_log_odds, conv_brick, conv_transpose_brick, bn_brick
# ...                 weights_init=IsotropicGaussian(),
# ...                 biases_init=Constant(0.01))
import numpy as np

class LeNet(FeedforwardSequence, Initializable):
    """LeNet-like convolutional network.

    The class implements LeNet, which is a convolutional sequence with
    an MLP on top (several fully-connected layers). For details see
    [LeCun95]_.

    .. [LeCun95] LeCun, Yann, et al.
       *Comparison of learning algorithms for handwritten digit
       recognition.*,
       International conference on artificial neural networks. Vol. 60.

    Parameters
    ----------
    conv_activations : list of :class:`.Brick`
        Activations for convolutional network.
    num_channels : int
        Number of channels in the input image.
    image_shape : tuple
        Input image shape.
    filter_sizes : list of tuples
        Filter sizes of :class:`.blocks.conv.ConvolutionalLayer`.
    feature_maps : list
        Number of filters for each of convolutions.
    pooling_sizes : list of tuples
        Sizes of max pooling for each convolutional layer.
    top_mlp_activations : list of :class:`.blocks.bricks.Activation`
        List of activations for the top MLP.
    top_mlp_dims : list
        Numbers of hidden units and the output dimension of the top MLP.
    conv_step : tuples
        Step of convolution (similar for all layers).
    border_mode : str
        Border mode of convolution (similar for all layers).

    """
    def __init__(self, conv_activations, num_channels, image_shape,
                 filter_sizes, feature_maps, pooling_sizes,
                 top_mlp_activations, top_mlp_dims,
                 conv_step=None, border_mode='valid', **kwargs):
        if conv_step is None:
            self.conv_step = (1, 1)
        else:
            self.conv_step = conv_step
        self.num_channels = num_channels
        self.image_shape = image_shape
        self.top_mlp_activations = top_mlp_activations
        self.top_mlp_dims = top_mlp_dims
        self.border_mode = border_mode

        conv_parameters = zip(filter_sizes, feature_maps)

        # Construct convolutional layers with corresponding parameters
        self.layers = list(interleave([
            (Convolutional(filter_size=filter_size,
                           num_filters=num_filter,
                           step=self.conv_step,
                           border_mode=self.border_mode,
                           name='conv_{}'.format(i))
             for i, (filter_size, num_filter)
             in enumerate(conv_parameters)),
            conv_activations,
            (MaxPooling(size, name='pool_{}'.format(i))
             for i, size in enumerate(pooling_sizes))]))

        self.conv_sequence = ConvolutionalSequence(self.layers, num_channels,
                                                   image_size=image_shape)

        # We need to flatten the output of the last convolutional layer.
        # This brick accepts a tensor of dimension (batch_size, ...) and
        # returns a matrix (batch_size, features)
        self.flattener = Flattener()

        # Construct a top MLP
        self.top_mlp = MLP(top_mlp_activations, top_mlp_dims)

        application_methods = [self.conv_sequence.apply, self.flattener.apply,
                               self.top_mlp.apply]
        super(LeNet, self).__init__(application_methods, **kwargs)

    @property
    def output_dim(self):
        return self.top_mlp_dims[-1]

    @output_dim.setter
    def output_dim(self, value):
        self.top_mlp_dims[-1] = value

    def _push_allocation_config(self):
        self.conv_sequence._push_allocation_config()
        conv_out_dim = self.conv_sequence.get_dim('output')

        self.top_mlp.activations = self.top_mlp_activations
        self.top_mlp.dims = [np.prod(conv_out_dim)] + self.top_mlp_dims


class Embedder(Initializable):
    """
    Linear Embedding Brick
    Parameters
    ----------
    dim_in: :class:`int`
        Dimensionality of the input
    dim_out: :class:`int`
        Dimensionality of the output
    output_type: :class:`str`
        fc for fully connected. conv for convolutional
    """

    def __init__(self, dim_in, dim_out, output_type='fc', **kwargs):

        self.dim_in = dim_in
        self.dim_out = dim_out
        self.output_type = output_type
        self.linear = Linear(dim_in, dim_out, name='embed_layer')
        children = [self.linear]
        kwargs.setdefault('children', []).extend(children)
        super(Embedder, self).__init__(**kwargs)

    @application(inputs=['y'], outputs=['outputs'])
    def apply(self, y):
        # embedding = self.linear.apply(y)
        embedding = y
        if self.output_type == 'fc':
            return embedding
        if self.output_type == 'conv':
            return embedding.reshape((-1, embedding.shape[-1], 1, 1))

    def get_dim(self, name):
        if self.output_type == 'fc':
            return self.linear.get_dim(name)
        if self.output_type == 'conv':
            return (self.linear.get_dim(name), 1, 1)


class EncoderMapping(Initializable):
    """
    Parameters
    ----------
    layers: :class:`list`
        list of bricks
    num_channels: :class: `int`
           Number of input channels
    image_size: :class:`tuple`
        Image size
    n_emb: :class:`int`
        Dimensionality of the embedding
    use_bias: :class:`bool`
        self explanatory
    """
    def __init__(self, layers, num_channels, image_size, n_emb, use_bias=False, **kwargs):
        self.layers = layers
        self.num_channels = num_channels
        self.image_size = image_size

        self.pre_encoder = ConvolutionalSequence(layers=layers[:-1],
                                                 num_channels=num_channels,
                                                 image_size=image_size,
                                                 use_bias=use_bias,
                                                 name='encoder_conv_mapping')
        self.pre_encoder.allocate()
        n_channels = n_emb + self.pre_encoder.get_dim('output')[0]
        self.post_encoder = ConvolutionalSequence(layers=[layers[-1]],
                                                  num_channels=n_channels,
                                                  image_size=(1, 1),
                                                  use_bias=use_bias)
        children = [self.pre_encoder, self.post_encoder]
        kwargs.setdefault('children', []).extend(children)
        super(EncoderMapping, self).__init__(**kwargs)

    @application(inputs=['x', 'y'], outputs=['output'])
    def apply(self, x, y):
        "Returns mu and logsigma"
        # Getting emebdding
        pre_z = self.pre_encoder.apply(x)
        # Concatenating
        pre_z_embed_y = tensor.concatenate([pre_z, y], axis=1)
        # propagating through last layer
        return self.post_encoder.apply(pre_z_embed_y)


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
                                             num_channels=num_channels,
                                             image_size=image_size,
                                             use_bias=use_bias,
                                             name='decoder_mapping0')
        self.mapping1 = ConvolutionalSequence(layers=layers[:-5],
                                             num_channels=num_channels,
                                             image_size=image_size,
                                             use_bias=use_bias,
                                             name='decoder_mapping1')
        self.mapping = ConvolutionalSequence(layers=layers[-5:],
                                             num_channels=128/16,
                                             image_size=(32,32),
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
        # z_y = ( self.mapping0.apply(z_y) + self.mapping1.apply(z_y) )
        return self.mapping.apply(z_y)


class GaussianConditional(Initializable, Random):
    def __init__(self, mapping, **kwargs):
        self.mapping = mapping
        super(GaussianConditional, self).__init__(**kwargs)
        self.children.extend([mapping])
    @property
    def _nlat(self):
        # if isinstance(self.mapping, ConvolutionalSequence):
        #     return self.get_dim('output')[0]
        # else:
        #     return self.get_dim('output')
        return self.mapping.children[-1].get_dim('output')[0] // 2

    def get_dim(self, name):
        if isinstance(self.mapping, ConvolutionalSequence):
            dim = self.mapping.get_dim(name)
            if name == 'output':
                return (dim[0] // 2) + dim[1:]
            else:
                return dim
        else:
            if name == 'output':
                # HACK: right way is not working
                return (256,)
                # return self.mapping.output_dim // 2
            elif name == 'input_':
                return self.mapping.input_dim
            else:
                return self.mapping.get_dim(name)
    @application(inputs=['x', 'y'], outputs=['output'])
    def apply(self, x, y, application_call):
        params = self.mapping.apply(x, y)
        mu, log_sigma = params[:, :self._nlat], params[:, self._nlat:]
        sigma = tensor.exp(log_sigma)
        epsilon = self.theano_rng.normal(size=mu.shape)
        return mu + sigma * epsilon


class XZYJointDiscriminator(Initializable):
    """Three-way discriminator.

    Parameters
    ----------
    x_discriminator : :class:`blocks.bricks.Brick`
        Part of the discriminator taking :math:`x` as input. Its
        output will be concatenated with ``z_discriminator``'s output
        and fed to ``joint_discriminator``.
    z_discriminator : :class:`blocks.bricks.Brick`
        Part of the discriminator taking :math:`z` as input. Its
        output will be concatenated with ``x_discriminator``'s output
        and fed to ``joint_discriminator``.
    joint_discriminator : :class:`blocks.bricks.Brick`
        Part of the discriminator taking the concatenation of
        ``x_discriminator``'s and output ``z_discriminator``'s output
        as input and computing :math:`D(x, z)`.

    """
    def __init__(self, x_discriminator, z_discriminator, joint_discriminator,
                 **kwargs):
        self.x_discriminator = x_discriminator # G_z(x,y)
        self.z_discriminator = z_discriminator # G_x(z,y)
        self.joint_discriminator = joint_discriminator

        super(XZYJointDiscriminator, self).__init__(**kwargs)
        self.children.extend([self.x_discriminator, self.z_discriminator,
                              self.joint_discriminator])

    @application(inputs=['x', 'z', 'y'], outputs=['output'])
    def apply(self, x, z, y):
        # NOTE: the unbroadcasts act as a workaround for a weird broadcasting
        # bug when applying dropout
        input_ = tensor.unbroadcast(
            tensor.concatenate(
                [self.x_discriminator.apply(x), self.z_discriminator.apply(z), y],
                axis=1),
            *range(x.ndim))
        return self.joint_discriminator.apply(input_)


class ConditionalALI(Initializable, Random):
    """Adversarial learned inference brick.

    Parameters
    ----------
    encoder : :class:`blocks.bricks.Brick`
        Encoder network.
    decoder : :class:`blocks.bricks.Brick`
        Decoder network.
    discriminator : :class:`blocks.bricks.Brick`
        Discriminator network taking :math:`x` and :math:`z` as input.
    n_cond: `int`
        Dimensionality of conditional data
    n_emb: `int`
        Dimensionality of embedding

    """
    def __init__(self, encoder, decoder, classifier, discriminator, n_cond, n_emb, **kwargs):
        self.encoder = encoder
        self.decoder = decoder
        self.discriminator = discriminator
        self.classifier = classifier
        self.n_cond = n_cond  # Features in conditional data
        self.n_emb = n_emb  # Features in embeddings
        self.embedder = Embedder(n_cond, n_emb, output_type='conv')

        super(ConditionalALI, self).__init__(**kwargs)
        self.children.extend([self.encoder, self.decoder, self.discriminator,
                              self.embedder, self.classifier])

    @property
    def discriminator_parameters(self):
        return list(
            Selector([self.discriminator]).get_parameters().values())

    @property
    def generator_parameters(self):
        return list(
            Selector([self.encoder, self.decoder]).get_parameters().values())
    @property
    def decoder_parameters(self):
        return list(
            Selector([self.decoder]).get_parameters().values())
    @property
    def embedding_parameters(self):
        return list(
            Selector([self.embedder]).get_parameters().values())

    @property
    def classifier_parameters(self):
        return list(
            Selector([self.classifier]).get_parameters().values())

    @application(inputs=['x', 'z_hat', 'x_tilde', 'z', 'y'],
                 outputs=['data_preds', 'sample_preds'])
    def get_predictions(self, x, z_hat, x_tilde, z, y, application_call):
        # NOTE: the unbroadcasts act as a workaround for a weird broadcasting
        # bug when applying dropout
        input_x = tensor.unbroadcast(
            tensor.concatenate([x, x_tilde], axis=0), *range(x.ndim))
        input_z = tensor.unbroadcast(
            tensor.concatenate([z_hat, z], axis=0), *range(x.ndim))
        input_y = tensor.unbroadcast(tensor.concatenate([y, y], axis=0), *range(x.ndim))
        data_sample_preds = self.discriminator.apply(input_x, input_z, input_y)
        data_preds = data_sample_preds[:x.shape[0]]
        sample_preds = data_sample_preds[x.shape[0]:]

        application_call.add_auxiliary_variable(
            tensor.nnet.sigmoid(data_preds).mean(), name='data_accuracy')
        application_call.add_auxiliary_variable(
            (1 - tensor.nnet.sigmoid(sample_preds)).mean(),
            name='sample_accuracy')

        return data_preds, sample_preds

    @application(inputs=['x', 'z', 'y'],
                 outputs=['discriminator_loss', 'generator_loss'])
    def compute_losses(self, x, z, y, application_call):
        embeddings = self.embedder.apply(y)
        z_hat = self.encoder.apply(x, embeddings)  # G_z(x,e(y))
        x_tilde = self.decoder.apply(z, embeddings) # G_x(z,e(y))

        data_preds, sample_preds = self.get_predictions(x, z_hat, x_tilde, z,
                                                        embeddings)
        # To be modularized
        # discriminator_loss = (tensor.nnet.softplus(-data_preds) +
        #                       tensor.nnet.softplus(sample_preds)).mean()
        # generator_loss = (tensor.nnet.softplus(data_preds) +
        #                   tensor.nnet.softplus(-sample_preds)).mean()

        discriminator_loss = (tensor.nnet.softplus(data_preds) +
                              tensor.nnet.softplus(-sample_preds)).mean()
        generator_loss = (tensor.nnet.softplus(-data_preds) +
                          tensor.nnet.softplus(sample_preds)).mean()

        return discriminator_loss, generator_loss

    @application(inputs=['z', 'y'], outputs=['samples'])
    def sample(self, z, y):
        return self.decoder.apply(z, self.embedder.apply(y))

    @application(inputs=['x', 'y'], outputs=['reconstructions'])
    def reconstruct(self, x, y):
        embeddings = self.embedder.apply(y)
        encoded = self.encoder.apply(x, embeddings)
        decoded = self.decoder.apply(encoded, embeddings)
        return decoded

    @application(inputs=['y'], outputs=['embeddings'])
    def embed(self, y):
        embeddings = self.embedder.apply(y)
        return embeddings

    @application(inputs=['x', 'y'], outputs=['encoded'])
    def encode(self, x, y):
        embeddings = self.embedder.apply(y)
        encoded = self.encoder.apply(x, embeddings)
        return encoded

    @application(inputs=['z', 'y'], outputs=['decoded'])
    def decode(self, z, y):
        embeddings = self.embedder.apply(y)
        decoded = self.decoder.apply(z, embeddings)
        return decoded

    @application(inputs=['z', 'e'], outputs=['decoded'])
    def decode_embedded(self, z, e):
        decoded = self.decoder.apply(z, e)
        return decoded

if __name__ == '__main__':
    import numpy as np
    import numpy.random as npr

    WEIGHTS_INIT = IsotropicGaussian(0.01)
    BIASES_INIT = Constant(0.)
    LEAK = 0.1
    NLAT = 64

    IMAGE_SIZE = (32, 32)
    NUM_CHANNELS = 3
    NUM_PIECES = 2

    NCLASSES = 10
    NEMB = 100
    # Testing embedder
    embedder = Embedder(NCLASSES, NEMB, output_type='conv',
                        weights_init=WEIGHTS_INIT, biases_init=BIASES_INIT)
    embedder.initialize()

    x = tensor.tensor4('x')
    y = tensor.matrix('y')

    embedder_test = function([y], embedder.apply(y))

    test_labels = np.zeros(shape=(5, 10))
    idx = npr.randint(0, 9, size=5)
    for n, id in enumerate(idx):
        test_labels[n, id] = 1
    embeddings = embedder_test(test_labels)
    print(embeddings)
    print(embeddings.shape)

    # Generate synthetic 4D tensor
    features = npr.random(size=(5, 3, 32, 32))

    # Testing Encoder
    layers = [
        # 32 X 32 X 3
        conv_brick(5, 1, 32), bn_brick(), LeakyRectifier(leak=LEAK),
        # 28 X 28 X 32
        conv_brick(4, 2, 64), bn_brick(), LeakyRectifier(leak=LEAK),
        # 13 X 13 X 64
        conv_brick(4, 1, 128), bn_brick(), LeakyRectifier(leak=LEAK),
        # 10 X 10 X 128
        conv_brick(4, 2, 256), bn_brick(), LeakyRectifier(leak=LEAK),
        # 4 X 4 X 256
        conv_brick(4, 1, 512), bn_brick(), LeakyRectifier(leak=LEAK),
        # 1 X 1 X 512
        conv_brick(1, 1, 512), bn_brick(), LeakyRectifier(leak=LEAK),
        # 1 X 1 X 512
        conv_brick(1, 1, 2 * NLAT)
        # 1 X 1 X 2 * NLAT
    ]

    encoder_mapping = EncoderMapping(layers=layers,
                                     num_channels=NUM_CHANNELS,
                                     image_size=IMAGE_SIZE, weights_init=WEIGHTS_INIT,
                                     biases_init=BIASES_INIT)
    encoder_mapping.initialize()

    embeddings = embedder.apply(y)
    encoder_mapping_fun = function([x, y], encoder_mapping.apply(x, embeddings))
    out = encoder_mapping_fun(features, test_labels)
    print(out.shape)

    ## Testing Gaussian encoder blocks
    embeddings = embedder.apply(y)
    encoder = GaussianConditional(mapping=encoder_mapping)
    encoder.initialize()
    encoder_fun = function([x, y], encoder.apply(x, embeddings))
    z_hat = encoder_fun(features, test_labels)
    # print(out)
    print(z_hat)

    # Decoder
    z = tensor.tensor4('z')
    layers = [
        conv_transpose_brick(4, 1, 256), bn_brick(), LeakyRectifier(leak=LEAK),
        conv_transpose_brick(4, 2, 128), bn_brick(), LeakyRectifier(leak=LEAK),
        conv_transpose_brick(4, 1, 64), bn_brick(), LeakyRectifier(leak=LEAK),
        conv_transpose_brick(4, 2, 32), bn_brick(), LeakyRectifier(leak=LEAK),
        conv_transpose_brick(5, 1, 32), bn_brick(), LeakyRectifier(leak=LEAK),
        conv_transpose_brick(1, 1, 32), bn_brick(), LeakyRectifier(leak=LEAK),
        conv_brick(1, 1, NUM_CHANNELS), Logistic()]

    decoder = Decoder(layers=layers, num_channels=(NLAT + NEMB), image_size=(1, 1),
                      weights_init=WEIGHTS_INIT, biases_init=BIASES_INIT)
    decoder.initialize()
    decoder_fun = function([z, y], decoder.apply(z, embeddings))
    out = decoder_fun(z_hat, test_labels)

    # Discriminator

    layers = [
        conv_brick(5, 1, 32), ConvMaxout(num_pieces=NUM_PIECES),
        conv_brick(4, 2, 64), ConvMaxout(num_pieces=NUM_PIECES),
        conv_brick(4, 1, 128), ConvMaxout(num_pieces=NUM_PIECES),
        conv_brick(4, 2, 256), ConvMaxout(num_pieces=NUM_PIECES),
        conv_brick(4, 1, 512), ConvMaxout(num_pieces=NUM_PIECES)]
    x_discriminator = ConvolutionalSequence(
        layers=layers, num_channels=NUM_CHANNELS, image_size=IMAGE_SIZE,
        name='x_discriminator')
    x_discriminator.push_allocation_config()

    layers = [
        conv_brick(1, 1, 512), ConvMaxout(num_pieces=NUM_PIECES),
        conv_brick(1, 1, 512), ConvMaxout(num_pieces=NUM_PIECES)]
    z_discriminator = ConvolutionalSequence(
        layers=layers, num_channels=NLAT, image_size=(1, 1), use_bias=False,
        name='z_discriminator')
    z_discriminator.push_allocation_config()

    layers = [
        conv_brick(1, 1, 1024), ConvMaxout(num_pieces=NUM_PIECES),
        conv_brick(1, 1, 1024), ConvMaxout(num_pieces=NUM_PIECES),
        conv_brick(1, 1, 1)]
    joint_discriminator = ConvolutionalSequence(
        layers=layers,
        num_channels=(x_discriminator.get_dim('output')[0] +
                      z_discriminator.get_dim('output')[0] +
                      NEMB),
        image_size=(1, 1),
        name='joint_discriminator')

    discriminator = XZYJointDiscriminator(
        x_discriminator, z_discriminator, joint_discriminator,
        name='discriminator')

    discriminator = XZYJointDiscriminator(x_discriminator, z_discriminator, joint_discriminator,
                                          name='discriminator', weights_init=WEIGHTS_INIT,
                                          biases_init=BIASES_INIT)
    discriminator.initialize()
    discriminator_fun = function([x, z, y], discriminator.apply(x, z, embeddings))
    out = discriminator_fun(features, z_hat, test_labels)
    print(out.shape)


    # Initializing ALI
    ali = ConditionalALI(encoder=encoder, decoder=decoder, discriminator=discriminator,
                         n_cond=NCLASSES,
                         n_emb=NEMB,
                         weights_init=WEIGHTS_INIT,
                         biases_init=BIASES_INIT)
    ali.initialize()
    # Computing Loss
    loss = ali.compute_losses(x, z, y)
    loss_fun = function([x, z, y], loss)
    out = loss_fun(features, z_hat, test_labels)


