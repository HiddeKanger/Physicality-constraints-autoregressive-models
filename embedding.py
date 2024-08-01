import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def positional_encoding(length, depth):
    depth = depth/2

    positions = np.arange(length)[:, np.newaxis]     # (seq, 1)
    depths = np.arange(depth)[np.newaxis, :]/depth   # (1, depth)

    angle_rates = 1 / (10000**depths)         # (1, depth)
    angle_rads = positions * angle_rates      # (pos, depth)

    pos_encoding = np.concatenate(
        [np.sin(angle_rads), np.cos(angle_rads)],
        axis=-1)

    return tf.cast(pos_encoding, dtype=tf.float32)

# Embedder used in transformer


class Embedder(tf.keras.layers.Layer):
    def __init__(self, vocab_size, d_model, n_qubits):
        super().__init__()
        self.d_model = d_model
        self.embedding = tf.keras.layers.Embedding(
            vocab_size, d_model, mask_zero=True)
        self.pos_encoding = positional_encoding(length=n_qubits, depth=d_model)
        print(self.pos_encoding.shape)

    def compute_mask(self, *args, **kwargs):
        return self.embedding.compute_mask(*args, **kwargs)

    def call(self, x):
        length = tf.shape(x)[1]
        x = self.embedding(x)
        # This factor sets the relative scale of the embedding and positonal_encoding.
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x = x + self.pos_encoding[tf.newaxis, :length, :]
        return x


if __name__ == "__main__":
    embedder = Embedder(
        vocab_size=4,  # 4 different POVM measurements
        d_model=16,  # hidden dimension of the model
        n_qubits=3,  # 2 qubits
    )

    input_tensor = tf.constant([[1, 2, 3], [2, 4, 3]], dtype=tf.float16)

    print(embedder(input_tensor))

    # one can see the length as the frequency and depth as the phase!
    # pos_encoding = positional_encoding(length=100, depth=100)

    # # Check the shape.
    # print(pos_encoding.shape)

    # # Plot the dimensions.
    # plt.pcolormesh(pos_encoding.numpy().T, cmap='RdBu')
    # plt.ylabel('Depth')
    # plt.xlabel('Position')
    # plt.colorbar()
    # plt.show()
