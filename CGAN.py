
from keras.optimizers import Adam
import os
import tensorflow.compat.v1 as tf
from keras.backend.tensorflow_backend import set_session
from PIL import Image
from tensorflow.python.keras import backend as K
import numpy as np

from keras import Model
from keras.initializers import RandomNormal, Zeros
from keras.layers import Input, ZeroPadding2D, Conv2D, BatchNormalization, Activation, Dropout, Add, Conv2DTranspose, LeakyReLU, Concatenate
from PIL import Image
import random


# -------------------------------------------------------- parameters
image_dir = './images/' # image path contains original dataset or dattaset after augmentation, it must contain two folders /train and /test
                                  # /train/a and /train/b, /test/a and /test/b  
direction = 'a2b'   # training direction from a2b (original image to segmented image) or b2a (segmented image to the original image): 
input_channel = 3  # input image channels
output_channel = 3  # output image channels
lr = 0.0002 # learning rate
epoch = 57
crop_from = 286
image_size = 256
batch_size = 1
combined_filepath = 'best_weights_before_augmentation.h5'    # or  'best_weights_after_augmentation.h5'  
generator_filepath = 'generator_before_augmentation.h5'      # or  'generator_after_augmentation.h5'     
seed = 9584
#----------------------------------------------------------- model


def residual(feature, dropout=False):
    x = Conv2D(256, kernel_size=3, strides=1, padding='same', kernel_initializer=RandomNormal(
        mean=0.0, stddev=0.02), bias_initializer=Zeros())(feature)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    if dropout:
        x = Dropout(0.5)(x)
    x = Conv2D(256, kernel_size=3, strides=1, padding='same', kernel_initializer=RandomNormal(
        mean=0.0, stddev=0.02), bias_initializer=Zeros())(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return Add()([feature, x])


def block_conv(feature, out_channel, downsample=True, dropout=False):
    if downsample:
        x = Conv2D(out_channel, kernel_size=4, strides=2, padding='same', kernel_initializer=RandomNormal(
            mean=0.0, stddev=0.02), bias_initializer=Zeros())(feature)
    else:
        x = Conv2DTranspose(out_channel, kernel_size=4, strides=2, padding='same', kernel_initializer=RandomNormal(
            mean=0.0, stddev=0.02), bias_initializer=Zeros())(feature)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    if dropout:
        x = Dropout(0.5)(x)
    return x




def generator_unet(n_block=3):
    input = Input(shape=(image_size, image_size, input_channel))
    # encoder
    encoder0 = Conv2D(64, kernel_size=4, padding='same', kernel_initializer=RandomNormal(
        mean=0.0, stddev=0.02), bias_initializer=Zeros())(input)  # use reflection padding instead
    encoder0 = BatchNormalization()(e0)
    encoder0 = Activation('relu')(encoder0)
    encoder1 = block_conv(encoder0, 128, downsample=True, dropout=False)  # 1/2
    encoder2 = block_conv(encoder1, 256, downsample=True, dropout=False)  # 1/4
    encoder3 = block_conv(encoder2, 512, downsample=True, dropout=False)  # 1/8
    encoder4 = block_conv(encoder3, 512, downsample=True, dropout=False)  # 1/16
    encoder5 = block_conv(encoder4, 512, downsample=True, dropout=False)  # 1/32
    encoder6 = block_conv(encoder5, 512, downsample=True, dropout=False)  # 1/64
    encoder7 = block_conv(encoder6, 512, downsample=True, dropout=False)  # 1/128
    # decoder
    decoder0 = block_conv(encoder7, 512, downsample=False, dropout=True)  # 1/64
    decoder1 = Concatenate(axis=-1)([decoder0, encoder6])
    decoder1 = block_conv(decoder1, 512, downsample=False, dropout=True)  # 1/32
    decoder2 = Concatenate(axis=-1)([decoder1, encoder5])
    decoder2 = block_conv(decoder2, 512, downsample=False, dropout=True)  # 1/16
    decoder3 = Concatenate(axis=-1)([decoder2, encoder4])
    decoder3 = block_conv(decoder3, 512, downsample=False, dropout=True)  # 1/8
    decoder4 = Concatenate(axis=-1)([decoder3, encoder3])
    decoder4 = block_conv(decoder4, 256, downsample=False, dropout=True)  # 1/4
    decoder5 = Concatenate(axis=-1)([decoder4, encoder2])
    decoder5 = block_conv(decoder5, 128, downsample=False, dropout=True)  # 1/2
    decoder6 = Concatenate(axis=-1)([decoder5, encoder1])
    decoder6 = block_conv(decoder6, 64, downsample=False, dropout=True)  # 1
    # out
    x = Conv2D(output_channel, kernel_size=3, padding='same', kernel_initializer=RandomNormal(
        mean=0.0, stddev=0.02), bias_initializer=Zeros())(d6)  # use reflection padding instead
    x = BatchNormalization()(x)
    x = Activation('tanh')(x)
    generator = Model(inputs=input, outputs=x)
    return generator


def generator_training_model(generator, discriminator):
    imgA = Input(shape=(image_size, image_size, input_channel))
    imgB = Input(shape=(image_size, image_size, input_channel))
    fakeB = generator(imgA)
    # discriminator.trainable=False
    realA_fakeB = Concatenate()([imgA, fakeB])
    pred_fake = discriminator(realA_fakeB)
    generator_training_model = Model(
        inputs=[imgA, imgB], outputs=[pred_fake, fakeB])
    return generator_training_model


def discriminator(n_layers=4, use_sigmoid=True):
    input = Input(shape=(image_size, image_size,
                         input_channel + output_channel))
    x = Conv2D(64, kernel_size=4, padding='same', strides=2, kernel_initializer=RandomNormal(
        mean=0.0, stddev=0.02), bias_initializer=Zeros())(input)
    x = LeakyReLU(alpha=0.2)(x)
    for i in range(1, n_layers):
        x = Conv2D(64 * 2 ** i, kernel_size=4, padding='same', strides=2, kernel_initializer=RandomNormal(
            mean=0.0, stddev=0.02), bias_initializer=Zeros())(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)
    x = Conv2D(64 * 2 ** n_layers, kernel_size=4, padding='same', strides=1, kernel_initializer=RandomNormal(
        mean=0.0, stddev=0.02), bias_initializer=Zeros())(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2D(1, kernel_size=4, padding='same', strides=1, kernel_initializer=RandomNormal(
        mean=0.0, stddev=0.02), bias_initializer=Zeros())(x)
    if use_sigmoid:
        x = Activation('sigmoid')(x)
    discriminator = Model(inputs=input, outputs=x)
    return discriminator

# -------------------------------------------------------------- image generator


def image_generator(a_path, b_path, batch_size, shuffle=True):
    image_filenames = os.listdir(a_path)
    n_batch = len(image_filenames) / batch_size if len(image_filenames) % batch_size == 0 else len(
        image_filenames) / batch_size + 1
    while True:
        if shuffle:
            random.shuffle(image_filenames)
        for i in range(int(n_batch)):
            a_batch = []
            b_batch = []
            for j in range(batch_size):
                index = i * batch_size + j
                if index >= len(image_filenames):
                    continue
                a = Image.open(os.path.join(
                    a_path, image_filenames[index])).convert('RGB')
                b = Image.open(os.path.join(
                    b_path, image_filenames[index])).convert('RGB')
                a = a.resize((crop_from, crop_from), Image.BICUBIC)
                b = b.resize((crop_from, crop_from), Image.BICUBIC)
                if random.random() < 0.5:
                    a = a.transpose(Image.FLIP_LEFT_RIGHT)
                    b = b.transpose(Image.FLIP_LEFT_RIGHT)
                a = np.asarray(a) / 127.5-1
                b = np.asarray(b) / 127.5-1
                w_offset = np.random.randint(0, max(
                    0, crop_from - image_size - 1)) if shuffle else (crop_from - image_size)/2
                h_offset = np.random.randint(0, max(
                    0, crop_from - image_size - 1)) if shuffle else (crop_from - image_size)/2
                a = a[int(h_offset):int(h_offset) + int(image_size),
                      int(w_offset):int(w_offset) + int(image_size), :]
                b = b[int(h_offset):int(h_offset) + int(image_size),
                      int(w_offset):int(w_offset) + int(image_size), :]
                a_batch.append(a)
                b_batch.append(b)
            if direction == 'a2b':
                yield np.array(a_batch), np.array(b_batch)
            else:
                yield np.array(b_batch), np.array(a_batch)
            # yield (np.array(a_batch) - imagenet_mean) / imagenet_std, (np.array(b_batch) - imagenet_mean) / imagenet_std


#-------------------------------------------------------------- train
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 1
config.gpu_options.allow_growth = True
K.set_session(tf.Session(config=config))

train_step_per_epoch = len(os.listdir(
    image_dir + 'train/a')) / batch_size + 1
print('--------------------------------------', train_step_per_epoch)
test_step_per_epoch = len(os.listdir(
    image_dir + 'test/a')) / batch_size + 1
print('---------------------------------------', train_step_per_epoch)
train_image_generator = image_generator(image_dir + 'train/a',
                                        image_dir + 'train/b', batch_size=batch_size,
                                        shuffle=True)
test_image_generator = image_generator(image_dir + 'test/a',
                                       image_dir + 'test/b', batch_size=batch_size,
                                       shuffle=False)

opt1 = Adam(lr=lr)
opt2 = Adam(lr=lr)
discriminator = discriminator()
print(discriminator.summary())
generator = generator_unet()
generator.compile(optimizer=opt2, loss='mae', metrics=[
                  'mean_absolute_percentage_error'])
print(generator.summary())
generator_train = generator_training_model(generator, discriminator)
print(generator_train.summary())
if os.path.exists(combined_filepath):
    generator_train.load_weights(combined_filepath, by_name=True)
    generator.load_weights(generator_filepath, by_name=True)
    print('weights loaded!')
discriminator.compile(optimizer=opt1, loss='mse', metrics=[
                      'acc'], loss_weights=None, sample_weight_mode=None)
generator_train.compile(optimizer=opt2, loss=['mse', 'mae'],
                        metrics=['mean_absolute_percentage_error'],
                        loss_weights=[1, 10])
real = np.ones((batch_size, 16, 16, 1))
fake = np.zeros((batch_size, 16, 16, 1))
best_loss = 1000

for i in range(epoch):
    train_step = 0
    for imgA, imgB in train_image_generator:
        train_step += 1
        if train_step > train_step_per_epoch:
            test_step = 0
            total_loss = 0
            total_mape = 0
            for imgA, imgB in test_image_generator:
                test_step += 1
                if test_step > test_step_per_epoch:
                    break
                gloss, mape = generator.test_on_batch(imgA, imgB)
                # print generator.metrics_names
                total_loss += gloss
                total_mape += mape
            print('epoch:{} test loss g:{:.2} \n   test mape:{}'.format(i + 1, total_loss / (test_step - 1),
                                                                        total_mape / (test_step - 1)))
            if total_loss / (test_step - 1) < best_loss:
                print('test loss improved from {} to {}'.format(
                    best_loss, total_loss / (test_step - 1)))
                generator_train.save_weights(combined_filepath, overwrite=True)
                generator.save_weights(generator_filepath, overwrite=True)
                best_loss = total_loss / (test_step - 1)
            break
        discriminator.trainable = True
        fakeB = generator.predict(imgA)


        fakeb = (fakeB[0] + 1) * 127.5
        fakeb = np.clip(fakeb, 0, 255)
        fakeb = fakeb.astype(np.uint8)
        fakeb = Image.fromarray(fakeb)
        fakeb.save('predict/' + str(i + 1) +
                    '_' + str(train_step) + '.png')
        print("{} saved".format('predict/' +
                                str(i + 1) + '_' + str(train_step) + '.png'))
        imgb = (imgB[0] + 1) * 127.5
        imgb = np.clip(imgb, 0, 255)
        imgb = imgb.astype(np.uint8)
        imgb = Image.fromarray(imgb)
        imgb.save('predict/' + str(i + 1) + '_' +
                    str(train_step) + '_real.png')
        print("{} saved".format('predict/' + str(i + 1) +
                                '_' + str(train_step) + '_real.png'))
        # print('realB:', imgB[0], imgB.shape)
        # print descriminator.trainable
        # print descriminator.summary()
        d_fake = discriminator.predict(
            np.concatenate((imgA, fakeB), axis=-1))
        d_real = discriminator.predict(
            np.concatenate((imgA, imgB), axis=-1))
        # print('d_real:', np.squeeze(d_real[0]), d_real.shape)
        # print('d_fake:', np.squeeze(d_fake[0]), d_fake.shape)
        loss_fake, fake_acc = discriminator.train_on_batch(
            np.concatenate((imgA, fakeB), axis=-1), fake)
        loss_real, real_acc = discriminator.train_on_batch(
            np.concatenate((imgA, imgB), axis=-1), real)
        print(
            'epoch:{} train step:{}, loss d_fake:{:.2}, loss d_real:{:.2}, fake_acc:{:.2}, real_acc:{:.2}'.format(i + 1, train_step,
                                                                                                                  loss_fake,
                                                                                                                  loss_real,
                                                                                                                  fake_acc,
                                                                                                                  real_acc))
        # print descriminator.metrics_names
        discriminator.trainable = False
        # print generator_train.summary()
        loss = generator_train.train_on_batch([imgA, imgB], [real, imgB])
        # print generator_train.metrics_names
        # print descriminator.trainable
        print('epoch:{} train step:{} loss fool:{:.2} loss g:{:.2}'.format(
            i + 1, train_step, loss[1], loss[0] - loss[1]))

