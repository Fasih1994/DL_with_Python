import keras
from keras import backend as K
from keras import layers
from keras.models import Model
import numpy as np


img_shape = (28, 28, 1)
batchSize = 16
latent_dims = 2


# encoder
input_img = layers.Input(shape=img_shape)
x = layers.Conv2D(32, 3, padding='same', activation='relu')(input_img)
x = layers.Conv2D(64, 3, strides=(2, 2), padding='same', activation='relu')(x)
x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)

shape_before_flattening = K.int_shape(x)
x = layers.Flatten()(x)
x = layers.Dense(32, activation='relu')(x)

z_mean = layers.Dense(latent_dims)(x)
z_log_varience = layers.Dense(latent_dims)(x)

# latent space sampling
def sampling(args):
    z_mean, z_log_varience = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0],latent_dims),
                              mean=0.0, stddev=1.0)
    return z_mean + K.exp(z_log_varience) * epsilon

# sampling layer
z = layers.Lambda(sampling)([z_mean,z_log_varience])

# VAE decoder network, mapping latent space points to images
decoder_input = layers.Input(K.int_shape(z)[1:])
x = layers.Dense(np.prod(shape_before_flattening[1:]), activation='relu')(decoder_input)
x = layers.Reshape(shape_before_flattening[1:])(x)
x = layers.Conv2DTranspose(32, 3,
                           padding='same',
                           strides=(2, 2),
                           activation='relu')(x)
x = layers.Conv2D(1, 3, padding='same', activation='sigmoid')(x)

decoder = Model(decoder_input, x)
z_decoded = decoder(z)

class CustomeVariationalLayers(layers.Layer):
    def vae_loss(self, x, z_decoded):
        x = K.flatten(x)
        z_decoded = K.flatten(z_decoded)
        xent_loss = keras.metrics.binary_crossentropy(x, z_decoded)
        kl_loss = -5e-3 * K.mean(1 + z_log_varience - K.square(z_mean) - K.exp(z_log_varience), axis=1)
        return K.mean(xent_loss + kl_loss)

    def call(self, inputs):
        x = inputs[0]
        z_decoded = inputs[1]
        loss = self.vae_loss(x, z_decoded)
        self.add_loss(losses=loss, inputs=inputs)
        return x

y = CustomeVariationalLayers()([input_img, z_decoded])



from keras.datasets import mnist

vae = Model(input_img, y)
vae.compile(optimizer='rmsprop', loss=None)
vae.summary()

(xtrain, _), (xtest, ytest) = mnist.load_data()
xtrain = xtrain.astype('float32') / 255.0
xtrain = xtrain.reshape(xtrain.shape + (1, ))
xtest = xtest.astype('float32') / 255.0
xtest = xtest.reshape(xtest.shape + (1, ))

vae.fit(x=xtrain, y=None,
        shuffle=True,
        epochs=10,
        batch_size=batchSize,
        validation_data=(xtest, None))

keras.models.save_model(decoder, 'decoder.h5')
keras.models.save_model(vae, 'mnist_VAE.h5')