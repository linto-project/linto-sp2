from keras.layers import Dense,Dropout, Activation, concatenate
from keras.regularizers import l2
from keras.models import Model

import models.resnet3d as m_r3d
import models.c3d as m_c3d
import generators.c3d as g_c3d
import generators.resnet3d as g_r3d
from models.i3d_inception import Inception_Inflated3d
import models.resnet2d as m_r2d
import models.vggvox as vggvox
import generators.vggvox as gen_vgg

doFlip  = True
doScale = True


def get_model(args):

    if args.net_audio == 'None':

        # model = c3d_model(num_classes, kernel_w, kernel_h)
        if 'resnet3D_18' == args.net_video:
            model, generator_train_batch, generator_val_batch, generator_test_batch = r3d_18(args.num_classes)
        elif 'resnet3D_34' == args.net_video:
            model, generator_train_batch, generator_val_batch, generator_test_batch = r3d_34(args.num_classes)
        elif 'resnet2D_concat' == args.net_video:
            model, generator_train_batch, generator_val_batch, generator_test_batch = r2d_50(args.num_classes)
        elif 'C3D' == args.net_video:
            model, generator_train_batch, generator_val_batch, generator_test_batch = c3d(args.num_classes)
        elif 'sC3D' == args.net_video:
            model, generator_train_batch, generator_val_batch, generator_test_batch = siamese_c3d(args.num_classes)
        elif 'sresnet3D_18' == args.net_video:
            model, generator_train_batch, generator_val_batch, generator_test_batch = siamese_r3d_18(args.num_classes)
        elif 'sresnet3D_34' == args.net_video:
            model, generator_train_batch, generator_val_batch, generator_test_batch = siamese_r3d_34(args.num_classes)
        else:
            raise Exception('No valid network.')
    else:
        model, generator_train_batch, generator_val_batch, generator_test_batch = video_audio(args=args)

    return generator_test_batch, generator_train_batch, generator_val_batch, model


def siamese_c3d(nb_classes, img_w=m_c3d.kernel_w, img_h=m_c3d.kernel_h, include_top=True):

    generator_train_batch = g_c3d.siamese_generator_train_batch
    generator_val_batch  =  g_c3d.siamese_generator_val_batch
    generator_test_batch =  g_c3d.siamese_generator_test_batch

    m_c3d.doFlip = doFlip
    m_c3d.doScale = doScale
    g_c3d.doFlip = doFlip
    g_c3d.doScale = doScale
    g_c3d.doNoisy = True
    g_c3d.doTrans = True

    model = m_c3d.siamese_c3d(nb_classes, img_w=img_w, img_h=img_h, include_top=include_top)
    return model, generator_train_batch, generator_val_batch, generator_test_batch


def c3d(nb_classes, img_w=m_c3d.kernel_w, img_h=m_c3d.kernel_h, include_top=True):
    # generator_train_batch = m_c3d.generator_train_batch
    # generator_val_batch  =  m_c3d.generator_val_batch
    # generator_test_batch =  m_c3d.generator_test_batch
    generator_train_batch = g_c3d.generator_train_batch
    generator_val_batch   = g_c3d.generator_val_batch
    generator_test_batch  = g_c3d.generator_test_batch

    # generator_train_batch = m_c3d.generator_train_batch_opt_flow
    # generator_val_batch  =  m_c3d.generator_val_batch_opt_flow
    # generator_test_batch =  m_c3d.generator_test_batch_opt_flow

    m_c3d.doFlip = doFlip
    m_c3d.doScale = doScale
    g_c3d.doFlip = doFlip
    g_c3d.doScale = doScale
    g_c3d.doNoisy = True
    g_c3d.doTrans = True

    return m_c3d.c3d_model(nb_classes=nb_classes, img_w=img_w, img_h=img_h, num_channels=3, include_top=include_top), generator_train_batch, generator_val_batch, generator_test_batch


def siamese_r3d_18(nb_classes, img_w=m_r3d.kernel_w, img_h=m_r3d.kernel_h, include_top=True):

    model = m_r3d.siamese_r3d_18(nb_classes, img_w, img_h, include_top=include_top)

    generator_train_batch = g_r3d.siamese_generator_train_batch
    generator_val_batch  =  g_r3d.siamese_generator_val_batch
    generator_test_batch =  g_r3d.siamese_generator_test_batch

    m_r3d.doFlip = doFlip
    m_r3d.doScale = doScale
    g_r3d.doFlip = doFlip
    g_r3d.doScale = doScale
    g_r3d.doNoisy = True
    g_r3d.doTrans = True

    return model, generator_train_batch, generator_val_batch, generator_test_batch


def siamese_r3d_34(nb_classes, img_w=m_r3d.kernel_w, img_h=m_r3d.kernel_h, include_top=True):

    model = m_r3d.siamese_r3d_34(nb_classes, img_w, img_h, include_top=include_top)

    generator_train_batch = g_r3d.siamese_generator_train_batch
    generator_val_batch  =  g_r3d.siamese_generator_val_batch
    generator_test_batch =  g_r3d.siamese_generator_test_batch

    m_r3d.doFlip = doFlip
    m_r3d.doScale = doScale
    g_r3d.doFlip = doFlip
    g_r3d.doScale = doScale
    g_r3d.doNoisy = True
    g_r3d.doTrans = True

    return model, generator_train_batch, generator_val_batch, generator_test_batch


def r3d_18(nb_classes, img_w=m_r3d.kernel_w, img_h=m_r3d.kernel_h, include_top=True):
    input_shape = (img_w, img_h, 16, 3)
    if include_top is False:
        nb_classes = -1
    model = m_r3d.Resnet3DBuilder.build_resnet_18(input_shape, nb_classes)
    generator_train_batch = g_r3d.generator_train_batch
    generator_val_batch  =  g_r3d.generator_val_batch
    generator_test_batch =  g_r3d.generator_test_batch

    m_r3d.doFlip = doFlip
    m_r3d.doScale = doScale
    g_r3d.doFlip = doFlip
    g_r3d.doScale = doScale
    g_r3d.doNoisy = True
    g_r3d.doTrans = True

    return model, generator_train_batch, generator_val_batch, generator_test_batch


def r3d_34(nb_classes, img_w=m_r3d.kernel_w, img_h=m_r3d.kernel_h, include_top=True):
    input_shape = (img_w, img_h, 16, 3)
    if include_top is False:
        nb_classes = -1
    model = m_r3d.Resnet3DBuilder.build_resnet_34(input_shape, nb_classes)
    generator_train_batch = g_r3d.generator_train_batch
    generator_val_batch   = g_r3d.generator_val_batch
    generator_test_batch  = g_r3d.generator_test_batch

    m_r3d.doFlip = doFlip
    m_r3d.doScale = doScale
    g_r3d.doFlip = doFlip
    g_r3d.doScale = doScale
    g_r3d.doNoisy = True
    g_r3d.doTrans = True

    return model, generator_train_batch, generator_val_batch, generator_test_batch


def r3d_50(nb_classes, img_w=m_r3d.kernel_w, img_h=m_r3d.kernel_h):
    input_shape = (img_w, img_h, 16, 3)
    model = m_r3d.Resnet3DBuilder.build_resnet_50(input_shape, nb_classes)
    generator_train_batch = m_r3d.generator_train_batch
    generator_val_batch =  m_r3d.generator_val_batch
    generator_test_batch =  m_r3d.generator_test_batch

    m_r3d.doFlip = doFlip
    m_r3d.doScale = doScale
    g_r3d.doFlip = doFlip
    g_r3d.doScale = doScale
    g_r3d.doNoisy = True
    g_r3d.doTrans = True

    return model, generator_train_batch, generator_val_batch, generator_test_batch


def i3d_inception(nb_classes, img_w=112, img_h=112):
    input_shape = (16, img_w, img_h, 3)
    model = Inception_Inflated3d(input_shape=input_shape, classes=nb_classes)
    return model


def r2d_50(nb_classes, img_w=m_r2d.kernel_w, img_h=m_r2d.kernel_h, clip_size=m_r2d.clip_size):
    model = m_r2d.resnet50_model(nb_classes, img_w, img_h, 3*clip_size)
    generator_train_batch = m_r2d.generator_train_batch
    generator_val_batch =  m_r2d.generator_val_batch
    generator_test_batch =  m_r2d.generator_test_batch

    m_r2d.doFlip = doFlip
    m_r2d.doScale = doScale

    return model, generator_train_batch, generator_val_batch, generator_test_batch


def video_audio(mode='train', args=None):
    input_audio = (257, 129, 1)
    num_class = args.num_classes


    if 'C3D' == args.net_video:
        x, generator_train_batch, generator_val_batch, generator_test_batch = \
            c3d(nb_classes=args.num_classes, include_top=False)
        gen_vgg.video_generator = g_c3d
        video_input = 'rgb'
    elif 'resnet3D_18' == args.net_video:
        x, generator_train_batch, generator_val_batch, generator_test_batch = \
            r3d_18(args.num_classes,include_top=False)
        gen_vgg.video_generator = g_r3d
        video_input = 'rgb'
    elif 'resnet3D_34' == args.net_video:
        x, generator_train_batch, generator_val_batch, generator_test_batch = \
            r3d_34(args.num_classes,include_top=False)
        gen_vgg.video_generator = g_r3d
        video_input = 'rgb'
    elif 'sC3D' == args.net_video:
        x, generator_train_batch, generator_val_batch, generator_test_batch = \
            siamese_c3d(nb_classes=args.num_classes, include_top=False)
        gen_vgg.video_generator = g_c3d
        video_input = 'rgb+optflow'
    elif 'sresnet3D_18' == args.net_video:
        x, generator_train_batch, generator_val_batch, generator_test_batch = \
            siamese_r3d_18(args.num_classes,include_top=False)
        gen_vgg.video_generator = g_r3d
        video_input = 'rgb+optflow'
    elif 'sresnet3D_34' == args.net_video:
        x, generator_train_batch, generator_val_batch, generator_test_batch = \
            siamese_r3d_34(args.num_classes,include_top=False)
        gen_vgg.video_generator = g_r3d
        video_input = 'rgb+optflow'
    else:
        raise Exception('No valid network.')

    # y = vggvox.vggvox_resnet2d(args=args, input_dim=input_audio, mode=mode, include_top=False)

    y = vggvox.vggvox_resnet2d(args=args, input_dim=input_audio, mode='eval', include_top=True)
    y.load_weights('./pre-trained-weights/resnet34_vlad8_ghost2_bdim512_deploy/weights.h5')
    y.layers.pop()
    y.layers.pop()
    # loop over all layers in the base model and freeze them so they will
    # *not* be updated during the first training process
    for layer in y.layers:
        layer.trainable = False

    if 'C3D' == args.net_video:
        weight_decay = 0.005
        z = concatenate([x.output, y.output])
        z = Dense(2048, activation='relu', kernel_regularizer=l2(weight_decay))(z)
        z = Dropout(0.5)(z)
        z = Dense(2048, activation='relu', kernel_regularizer=l2(weight_decay))(z)
        z = Dropout(0.5)(z)
        z = Dense(num_class, kernel_regularizer=l2(weight_decay))(z)
        z = Activation('softmax')(z)
    else:
        z = concatenate([x.output, y.output])
        z = Dense(units=num_class,
                  kernel_initializer="he_normal",
                  activation="softmax",
                  kernel_regularizer=l2(1e-4))(z)

    m_c3d.doFlip = doFlip
    m_c3d.doScale = doScale
    gen_vgg.video_generator.doFlip = True
    gen_vgg.video_generator.doScale = False
    gen_vgg.video_generator.doNoisy = True
    gen_vgg.video_generator.doTrans = True

    if video_input == 'rgb':
        generator_train_batch = gen_vgg.generator_train_batch_rgb_audio
        generator_val_batch   = gen_vgg.generator_val_batch_rgb_audio
        generator_test_batch  = gen_vgg.generator_test_batch_rgb_audio
        model = Model(inputs=[x.input, y.input], outputs=z)
    elif video_input == 'rgb+optflow':
        generator_train_batch = gen_vgg.generator_train_batch_rgb_optflow_audio
        generator_val_batch   = gen_vgg.generator_val_batch_rgb_optflow_audio
        generator_test_batch  = gen_vgg.generator_test_batch_rgb_optflow_audio
        model = Model(inputs=[x.input[0], x.input[1], y.input], outputs=z)
    else:
        raise Exception('No valid network.')

    return model, generator_train_batch, generator_val_batch, generator_test_batch

    # model = vggvox.video_audio(input_audio=input_audio, num_class=num_class, mode=mode, args=args)
    # return model, gen_vgg.generator_train_batch_rgb_audio, gen_vgg.generator_val_batch_rgb_audio, gen_vgg.generator_test_batch_rgb_audio
