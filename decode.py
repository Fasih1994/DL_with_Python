from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

batch_size = 16
n = 15
digit_size = 28
figure = np.zeros((digit_size * n, digit_size * n))
grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
grid_y = norm.ppf(np.linspace(0.05, 0.95, n))

decoder = load_model('decoder.h5')

for i, yi in enumerate(grid_y):
    for j, xj in enumerate(grid_x):
        z_sample = np.array([[xj, yi]])
        z_sample = np.tile(z_sample, batch_size).reshape(batch_size, 2)
        x_decoded = decoder.predict(z_sample, batch_size=batch_size)
        digit = x_decoded[0].reshape(digit_size, digit_size)
        figure[i*digit_size: (i+1) * digit_size, j * digit_size: (j+1) * digit_size] = digit

plt.figure(figsize=(10, 10))
plt.imshow(figure, cmap='Greys_r')
plt.show()
plt.imsave('mnist.jpg', figure, cmap='Greys_r')