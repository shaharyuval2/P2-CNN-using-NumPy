import numpy as np

"""
batch_size = 3
c_in = 2
c_out = 4
Ih = 5
Iw = 5
I_batch_values = np.arange(batch_size * c_in * Ih * Iw)
I_batch = I_batch_values.reshape(batch_size, c_in, Ih, Iw)

Kh = 3
Kw = 3
K = np.ones((c_in, c_out, Kh, Kw))

Oh = Ih - Kh + 1
Ow = Iw - Kw + 1

O_batch = np.zeros((batch_size, c_out, Oh, Ow))

for q in range(c_out):
    for p in range(c_in):
        # passes into 3d correlate I_p = (batch_size,Ih,Iw) , K_p,q = (1,Kh,Kw)
        O_batch[:, q] += signal.correlate(
            I_batch[:, p, :, :], K[p, q, np.newaxis, :, :], mode="valid"
        )

print(O_batch)
"""

# test @
weights = np.ones((3, 5))
biases = np.ones((3, 1)) * 1000
inputs_range = np.arange(50)
inputs = inputs_range.reshape((10, 5, 1))

print(weights @ inputs + biases)

# test squeeze
y_hat = np.ones((10, 3, 1))
print(y_hat)
print(y_hat.squeeze(axis=2))
