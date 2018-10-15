import model
import numpy as np
import pickle
import image_utils as imgu
import matplotlib.pyplot as plt
from keras.layers import Input
from keras.models import Model
from keras.optimizers import RMSprop

image_size = (128, 128)
fake_shape = (128, 128, 1)
input_shape = (128, 128, 3)

epochs = 500
discriminator_epochs = 2
batch_size = 64
ngf = 16

generator = model.generator(ngf, fake_shape)
discriminator = model.discriminator(ngf, input_shape)

optimizer = RMSprop(lr=.01)

# compile the discriminator
discriminator.compile(loss=model.W_loss, optimizer='adam', metrics=['accuracy'])

# compile the generator
generator.compile(loss='binary_crossentropy', optimizer='adam')

# generate fake image from prior and noise
z = Input(shape=(ngf * 8, ))
x = Input(shape=fake_shape)
y = Input(shape=input_shape)

fake_img = generator([x, z])

# The valid takes generated images as input and determines validity
valid = discriminator([y, fake_img])

# The combined model  (stacked generator and discriminator) takes
# noise as input => generates images => determines validity
combined = Model([x, z, y], [valid, fake_img])
combined.compile(loss=['binary_crossentropy', 'mean_absolute_error'], loss_weights=[1, .01], optimizer='adam')

#####################################################################

dis_loss = []
gen_loss = []


for epoch in range(epochs):

    # load up the images
    if epoch % 40 == 0:

        idx = np.random.randint(0, 10)
        last_idx = idx

        while idx == last_idx:
            idx = np.random.randint(0, 10)

        # edge_img = imgu.image_generator_index('edges/', index=idx)
        # target_img = imgu.image_generator_index('resize/', index=idx)

        edge_img = pickle.load(open('images_pickle/edges_{}.pkl'.format(idx), 'rb'))[..., np.newaxis]
        target_img = pickle.load(open('images_pickle/img_{}.pkl'.format(idx), 'rb'))

        edge_img = (edge_img - 127.5) / 127.5
        target_img = (target_img - 127.5) / 127.5

    steps = len(target_img)//batch_size

    for step in range(steps):
        #random label smoothing
        positive_y = np.random.choice([0, 1], size=batch_size, p=[.25, .75])
        negative_y = np.random.choice([0, 1], size=batch_size, p=[.75, .25])
        #
        # positive_y = np.clip(np.random.normal(.8, .2, size=batch_size), 0, 1)
        # negative_y = np.clip(np.random.normal(.3, .2, size=batch_size), 0, 1)

        idx = np.random.randint(0, len(target_img), batch_size)

        noise = np.random.normal(0, 1, (batch_size, ngf * 8))

        fake = edge_img[idx]
        real = target_img[idx]

        # ---------------------
        #  Train Discriminator
        # ---------------------

        # prediction of the generator
        gen_img = generator.predict([fake, noise])

        # Train the discriminator

        for i in range(discriminator_epochs):
            # d_loss_real = discriminator.train_on_batch([real, real], np.ones((batch_size, 1)))
            # d_loss_fake = discriminator.train_on_batch([real, gen_img], np.zeros((batch_size, 1)))

            # train on W_loss
            d_loss_real = discriminator.train_on_batch([real, real], positive_y)
            d_loss_fake = discriminator.train_on_batch([real, gen_img], negative_y)

            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # ---------------------
        #  Train Generator
        # ---------------------

        # The generator wants the discriminator to label the generated samples
        # as valid (ones)


        # Train the generator
        # g_loss = combined.train_on_batch([fake, noise, real], [valid_y, real])


        # Train on W_loss

        positive_y = np.random.choice([0, 1], size=batch_size, p=[.15, .85])

        discriminator.trainable = False
        g_loss = combined.train_on_batch([fake, noise, real], [positive_y, real])
        discriminator.trainable = True

        # Plot the progress
        print("%d [D loss: %f, accuracy: %f] [G loss: %f, recon: %f]" % (epoch, d_loss[0], d_loss[1], g_loss[1], g_loss[2]))

        dis_loss.append(d_loss[0])
        gen_loss.append(d_loss[1])


def pow(x, n):
    if n == 0:
        return 1
    elif n == 1:
        return x
    else:
        if n % 2 == 0:
            temp = pow(x, n//2)

            result = temp * temp
        else:
            temp = pow(x, (n-1)//2)

            result = x * temp * temp

    return result

def fibonacci(n):
    """
    calculate the n-th fibonacci number
    :param n:
    :return:
    """

    if n == 0:
        return 0

    else:

        phi = 1.61803
        psi = -.61803

        F = (pow(phi, n) + pow(psi, n))/2.23606

        return round(F)
