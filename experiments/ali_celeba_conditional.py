import argparse
import logging

from blocks.algorithms import Adam, RMSProp, Momentum
from blocks.bricks import LeakyRectifier, Logistic, Rectifier, Softmax, Activation
from blocks.bricks.conv import ConvolutionalSequence
from blocks.extensions import FinishAfter, Timing, Printing, ProgressBar
from blocks.extensions.monitoring import DataStreamMonitoring
from blocks.extensions.saveload import Checkpoint
from blocks.filter import VariableFilter
from blocks.graph import ComputationGraph, apply_dropout
from blocks.graph.bn import (batch_normalization,
                             get_batch_normalization_updates)
from blocks.initialization import IsotropicGaussian, Constant, Uniform
from blocks.main_loop import MainLoop
from blocks.model import Model
from blocks.roles import INPUT
from theano import tensor, grad

from ali.algorithms import ali_algorithm
from ali.conditional_bricks import (EncoderMapping, Decoder,
                                    GaussianConditional, XZYJointDiscriminator,
                                    ConditionalALI, LeNet)
from ali.streams import create_celeba_data_streams, create_crs_data_streams, create_mnist64_data_streams
from ali.utils import get_log_odds, conv_brick, conv_transpose_brick, bn_brick
from blocks.algorithms import GradientDescent
RATIO = 16
NCLASSES = 2

BATCH_SIZE = 128
MONITORING_BATCH_SIZE = 128
NUM_EPOCHS = 123
IMAGE_SIZE = (64, 64)
NUM_CHANNELS = 1
NLAT = 256/RATIO
NEMB = 256/RATIO

GAUSSIAN_INIT = IsotropicGaussian(std=0.01)
ZERO_INIT = Constant(0)
LEARNING_RATE_C = 1e-3
LEARNING_RATE_D = 1e-1
LEARNING_RATE_G = 1e-2
BETA1 = 0.5
LEAK = 0.02


def create_model_brick():
    # Encoder
    enc_layers = [
        conv_brick(2, 1, 64/RATIO), bn_brick(), LeakyRectifier(leak=LEAK),
        conv_brick(7, 2, 128/RATIO), bn_brick(), LeakyRectifier(leak=LEAK),
        conv_brick(5, 2, 256/RATIO), bn_brick(), LeakyRectifier(leak=LEAK),
        conv_brick(7, 2, 256/RATIO), bn_brick(), LeakyRectifier(leak=LEAK),
        conv_brick(4, 1, 512/RATIO), bn_brick(), LeakyRectifier(leak=LEAK),
        conv_brick(1, 1, 2 * NLAT)]

    encoder_mapping = EncoderMapping(layers=enc_layers,
                                     num_channels=NUM_CHANNELS,
                                     n_emb=NEMB,
                                     image_size=IMAGE_SIZE, weights_init=GAUSSIAN_INIT,
                                     biases_init=ZERO_INIT,
                                     use_bias=False)

    encoder = GaussianConditional(encoder_mapping, name='encoder')
    # Decoder
    dec_layers = [
        conv_transpose_brick(4, 1, 512/RATIO), bn_brick(), LeakyRectifier(leak=LEAK),
        conv_transpose_brick(7, 2, 256/RATIO), bn_brick(), LeakyRectifier(leak=LEAK),
        conv_transpose_brick(5, 2, 256/RATIO), bn_brick(), LeakyRectifier(leak=LEAK),
        conv_transpose_brick(7, 2, 128/RATIO), bn_brick(), LeakyRectifier(leak=LEAK),
        conv_transpose_brick(2, 1, 64/RATIO), bn_brick(), LeakyRectifier(leak=LEAK),
        conv_brick(1, 1, NUM_CHANNELS), Logistic()]

    decoder = Decoder(
        layers=dec_layers, num_channels=NLAT + NEMB, image_size=(1, 1), use_bias=False,
        name='decoder_mapping')
    # Discriminator for x
    layers = [
        conv_brick(2, 1, 64/RATIO), LeakyRectifier(leak=LEAK),
        conv_brick(7, 2, 128/RATIO), bn_brick(), LeakyRectifier(leak=LEAK),
        conv_brick(5, 2, 256/RATIO), bn_brick(), LeakyRectifier(leak=LEAK),
        conv_brick(7, 2, 256/RATIO), bn_brick(), LeakyRectifier(leak=LEAK),
        conv_brick(4, 1, 512/RATIO), bn_brick(), LeakyRectifier(leak=LEAK)]
    x_discriminator = ConvolutionalSequence(
        layers=layers, num_channels=NUM_CHANNELS, image_size=IMAGE_SIZE,
        use_bias=False, name='x_discriminator')
    x_discriminator.push_allocation_config()
    # Discriminator for z
    layers = [
        conv_brick(1, 1, 1024/RATIO), LeakyRectifier(leak=LEAK),
        conv_brick(1, 1, 1024/RATIO), LeakyRectifier(leak=LEAK)]
    z_discriminator = ConvolutionalSequence(
        layers=layers, num_channels=NLAT, image_size=(1, 1), use_bias=False,
        name='z_discriminator')
    z_discriminator.push_allocation_config()
    # Discriminator for joint
    layers = [
        conv_brick(1, 1, 2048/RATIO), LeakyRectifier(leak=LEAK),
        conv_brick(1, 1, 2048/RATIO), LeakyRectifier(leak=LEAK),
        conv_brick(1, 1, 1)]
    joint_discriminator = ConvolutionalSequence(
        layers=layers,
        num_channels=(x_discriminator.get_dim('output')[0] +
                      z_discriminator.get_dim('output')[0] +
                      NEMB),
        image_size=(1, 1),
        name='joint_discriminator')
    # D( x, z, y )
    discriminator = XZYJointDiscriminator(
        x_discriminator, z_discriminator, joint_discriminator,
        name='discriminator')

    feature_maps = [16, 32, 64]
    mlp_hiddens = [200]
    output_size = NCLASSES
    image_size = (64, 64)

    conv_activations = [Rectifier() for _ in feature_maps]
    mlp_activations = [Rectifier() for _ in mlp_hiddens] + [Logistic()]
    convnet = LeNet(conv_activations, 1, image_size,
                    filter_sizes=[(3, 3), (3, 3), (3, 3)],
                    feature_maps=feature_maps,
                    pooling_sizes=[(2, 2), (2, 2), (2, 2)],
                    top_mlp_activations=mlp_activations,
                    top_mlp_dims=mlp_hiddens + [output_size],
                    border_mode='valid',
                    weights_init=Uniform(width=.5),
                    biases_init=Constant(0)
                    )

    convnet.push_initialization_config()
    # convnet.layers[0].weights_init = Uniform(width=.2)
    # convnet.layers[3].weights_init = Uniform(width=.2)
    # convnet.layers[6].weights_init = Uniform(width=.2)
    # convnet.layers[9].weights_init = Uniform(width=.2)
    # convnet.top_mlp.linear_transformations[0].weights_init = Uniform(width=.08)
    # convnet.top_mlp.linear_transformations[1].weights_init = Uniform(width=.08)
    # convnet.layers[0].biases_init = Uniform(width=.2)
    # convnet.layers[3].biases_init = Uniform(width=.2)
    # convnet.layers[6].biases_init = Uniform(width=.2)
    # convnet.layers[9].biases_init = Uniform(width=.2)
    # convnet.top_mlp.linear_transformations[0].biases_init = Uniform(width=.08)
    # convnet.top_mlp.linear_transformations[1].biases_init = Uniform(width=.08)
    convnet.initialize()

    logging.info("Input dim: {} {} {} ".format(
        *convnet.children[0].get_dim('input_')))
    for i, layer in enumerate(convnet.layers):
        if isinstance(layer, Activation):
            logging.info("Layer {} ({})".format(
                i, layer.__class__.__name__))
        else:
            logging.info("Layer {} ({}) dim: {} {} {}".format(
                i, layer.__class__.__name__, *layer.get_dim('output')))
    classifier = convnet

    ali = ConditionalALI(encoder, decoder, classifier, discriminator,
                         n_cond=NCLASSES, n_emb=NEMB,
                         weights_init=GAUSSIAN_INIT, biases_init=ZERO_INIT,
                         name='ali')
    ali.push_allocation_config()
    encoder_mapping.layers[-1].use_bias = True
    encoder_mapping.layers[-1].tied_biases = False
    decoder.layers[-2].use_bias = True
    decoder.layers[-2].tied_biases = False
    x_discriminator.layers[0].use_bias = True
    x_discriminator.layers[0].tied_biases = True
    ali.initialize()
    raw_marginals, = next(
        create_crs_data_streams(500, 500)[0].get_epoch_iterator())
    b_value = get_log_odds(raw_marginals)
    decoder.layers[-2].b.set_value(b_value)

    return ali


def create_models():
    ali = create_model_brick()
    x = tensor.tensor4('features')
    y = tensor.matrix('targets')
    z = ali.theano_rng.normal(size=(x.shape[0], NLAT, 1, 1))

    def _create_model(with_dropout):
        ls = ali.compute_losses(x, z, y)
        cg = ComputationGraph(ls)
        if with_dropout:
            inputs = VariableFilter(
                bricks=([ali.discriminator.x_discriminator.layers[0]] +
                        ali.discriminator.x_discriminator.layers[2::3] +
                        ali.discriminator.z_discriminator.layers[::2] +
                        ali.discriminator.joint_discriminator.layers[::2]),
                roles=[INPUT])(cg.variables)
            cg = apply_dropout(cg, inputs, 0.2)
        return Model(cg.outputs)

    model = _create_model(with_dropout=False)
    with batch_normalization(ali):
        bn_model = _create_model(with_dropout=True)

    pop_updates = list(
        set(get_batch_normalization_updates(bn_model, allow_duplicates=True)))
    bn_updates = [(p, m * 0.05 + p * 0.95) for p, m in pop_updates]

    from blocks.bricks.cost import CategoricalCrossEntropy, MisclassificationRate

    pred = ali.classifier.apply(x)
    classifier_cost = - ( tensor.sum(y*tensor.log(pred)) + tensor.sum((1-y)*tensor.log(1-pred)) )
    classifier_error = tensor.sum(tensor.sqr(y - pred))/BATCH_SIZE/NCLASSES

    # pred = tensor.reshape(ali.classifier.apply(x),(BATCH_SIZE,NUM_CHANNELS))
    # classifier_cost = (CategoricalCrossEntropy().apply(y.flatten(), pred.flatten()).copy(name='cost'))
    # classifier_error = (MisclassificationRate().apply(tensor.reshape(y,(BATCH_SIZE,NUM_CHANNELS)).flatten(), tensor.reshape(pred,(BATCH_SIZE,NUM_CHANNELS))).copy(name='error_rate'))

    # pred = ali.classifier.apply(x)
    # classifier_cost = tensor.nnet.categorical_crossentropy(pred, y).mean()
    # classifier_error = (MisclassificationRate().apply(y.flatten(), pred)
    #               .copy(name='error_rate'))

    classifier_cost.name = 'classifier cost'
    classifier_error.name = 'classifier error'

    embeddings = ali.embedder.apply(y)
    z_hat = ali.encoder.apply(x, embeddings)  # G_z(x,e(y))
    x_hat = ali.decoder.apply(z_hat, embeddings)  # G_x(z,e(y))
    y_hat = ali.classifier.apply(x_hat)

    eps = 1e-10
    c_cost = - ( tensor.sum(y*tensor.log(y_hat+eps)) + tensor.sum((1-y)*tensor.log(1-y_hat+eps)) )
    c_er = tensor.sum(tensor.sqr(y - y_hat))/BATCH_SIZE/2.
    c_cost.name = 'c_cost'
    c_er.name = 'c_er'
    return model, bn_model, bn_updates, classifier_cost, classifier_error, c_cost, c_er


def create_main_loop(save_path):
    model, bn_model, bn_updates, classifier_cost, classifier_error, c_cost, c_er= create_models()
    ali, = bn_model.top_bricks
    discriminator_loss, generator_loss = bn_model.outputs

    step_rule_c = RMSProp(learning_rate=LEARNING_RATE_C)
    # step_rule_g = Adam(learning_rate=LEARNING_RATE_G, beta1=BETA1)
    step_rule_g = Momentum(learning_rate=LEARNING_RATE_G,momentum = 0.99)
    step_rule_d = Momentum(learning_rate=LEARNING_RATE_D,momentum = 0.99)
    algorithm = ali_algorithm(discriminator_loss, ali.discriminator_parameters,
                              step_rule_d, generator_loss,
                              ali.generator_parameters, step_rule_g,
                              c_cost, step_rule_g)
    algorithm.add_updates(bn_updates)
    streams = create_crs_data_streams(BATCH_SIZE, MONITORING_BATCH_SIZE,
                                         sources=('features', 'targets'))
    main_loop_stream, train_monitor_stream, valid_monitor_stream = streams
    bn_monitored_variables = (
        [v for v in bn_model.auxiliary_variables if 'norm' not in v.name] +
        bn_model.outputs + [c_cost, c_er])
    monitored_variables = (
        [v for v in model.auxiliary_variables if 'norm' not in v.name] +
        model.outputs+ [c_cost, c_er])
    from blocks.monitoring import aggregation
    bn_monitored_variables += [algorithm.total_gradient_norm]
    extensions = [
        Timing(),
        FinishAfter(after_n_epochs=NUM_EPOCHS),
        DataStreamMonitoring(
            bn_monitored_variables, train_monitor_stream, prefix="train",
            updates=bn_updates),
        DataStreamMonitoring(
            monitored_variables, valid_monitor_stream, prefix="valid"),
        Checkpoint(save_path, after_epoch=True, after_training=True,
                   use_cpickle=True),
        ProgressBar(),
        Printing(),
    ]
    main_loop = MainLoop(model=bn_model, data_stream=main_loop_stream,
                         algorithm=algorithm, extensions=extensions)

    from blocks.algorithms import GradientDescent, CompositeRule, Restrict
    from collections import OrderedDict
    gradients = OrderedDict()
    gradients.update(
        zip(ali.classifier_parameters,
            grad(classifier_cost, ali.classifier_parameters)))
    classify_algorithm = GradientDescent(cost=classifier_cost,
                                        gradients=gradients,
                                        parameters=ali.classifier_parameters,
                                        step_rule=step_rule_c)
    classifier_monitor = ([classifier_cost, classifier_error])
    extensions = [
        Timing(),
        FinishAfter(after_n_epochs=5),
        DataStreamMonitoring(
            classifier_monitor, train_monitor_stream, prefix="train"),
        DataStreamMonitoring(
            classifier_monitor, valid_monitor_stream, prefix="valid"),
        Checkpoint('./testC.ckpt', after_epoch=True, after_training=True,
                   use_cpickle=True),
        ProgressBar(),
        Printing(),
    ]
    classify_loop = MainLoop(data_stream=main_loop_stream,
                             model=Model(classifier_cost),
                             algorithm=classify_algorithm,
                             extensions=extensions)
    print('classifier training...')
    classify_loop.run()
    print('classifier training done...')
    return main_loop


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Train ALI on CelebA")
    parser.add_argument("--save-path", type=str, default='ali_conditional_celeba.tar',
                        help="main loop save path")
    args = parser.parse_args()
    create_main_loop(args.save_path).run()
