import tensorflow as tf
import numpy as np

from utils.utils import read_config

# Sine/cosine embedding as in original paper


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
            vocab_size, d_model)
        self.pos_encoding = positional_encoding(
            length=n_qubits, depth=d_model)

    def compute_mask(self, *args, **kwargs):
        return self.embedding.compute_mask(*args, **kwargs)

    def call(self, x, pos_phase=0):
        length = tf.shape(x)[1]
        x = self.embedding(x)
        # This factor sets the relative scale of the embedding and positonal_encoding.
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))

        p_enc = tf.roll(input=self.pos_encoding, shift=pos_phase, axis=0)
        # print("p_enc: ", p_enc)

        # print("self.pos_encoding: ", self.pos_encoding)
        # x = x + self.pos_encoding[tf.newaxis, :length, :]
        x = x + p_enc[tf.newaxis, :length, :]
        return x

# Multihead attention + ADD/NORM base implementation


class BaseAttention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(**kwargs)
        self.layernorm = tf.keras.layers.LayerNormalization()
        self.add = tf.keras.layers.Add()

# Causaul -> apply mask


class CausalSelfAttention(BaseAttention):
    def call(self, x):
        attn_output = self.mha(
            query=x,
            value=x,
            key=x,
            use_causal_mask=True)
        x = self.add([x, attn_output])  # residual connection
        x = self.layernorm(x)
        return x

# Non causal variant -> use all correlations and not masked


class GlobalSelfAttention(BaseAttention):
    def call(self, x):
        attn_output = self.mha(
            query=x,
            value=x,
            key=x)
        x = self.add([x, attn_output])  # residual connection
        x = self.layernorm(x)
        return x

# Simple feedforward network


class FeedForward(tf.keras.layers.Layer):
    def __init__(self, d_model, dff, dropout_rate=0.1):
        super().__init__()
        self.seq = tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation='relu'),
            tf.keras.layers.Dense(d_model),
            tf.keras.layers.Dropout(dropout_rate)
        ])
        self.add = tf.keras.layers.Add()
        self.layer_norm = tf.keras.layers.LayerNormalization()

    def call(self, x, training):
        x = self.add([x, self.seq(x, training=training)])
        x = self.layer_norm(x)
        return x

# One decoder block (yellow block)


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self,
                 *,
                 d_model,
                 num_heads,
                 dff,
                 dropout_rate=0.1):
        super(DecoderLayer, self).__init__()

        self.causal_self_attention = CausalSelfAttention(
            num_heads=num_heads,
            key_dim=d_model,
            dropout=dropout_rate)

        self.ffn = FeedForward(d_model, dff)

    def call(self, x, training):
        x = self.causal_self_attention(x=x, training=training)

        # Cache the last attention scores for plotting later
        # self.last_attn_scores = self.cross_attention.last_attn_scores

        # Shape `(batch_size, seq_len, d_model)`.
        x = self.ffn(x=x, training=training)
        return x

# The decoder structure (yellow block, can be multiple)


class Decoder(tf.keras.layers.Layer):
    def __init__(self, *, num_layers, d_model, num_heads, dff, vocab_size, n_qubits,
                 dropout_rate=0.0):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.pos_embedding = Embedder(vocab_size=vocab_size,
                                      d_model=d_model,
                                      n_qubits=n_qubits)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.dec_layers = [
            DecoderLayer(d_model=d_model, num_heads=num_heads,
                         dff=dff, dropout_rate=dropout_rate)
            for _ in range(num_layers)]

        self.last_attn_scores = None

    def call(self, x, training, pos_phase=0):
        # `x` is token-IDs shape (batch, target_seq_len)
        # (batch_size, target_seq_len, target_vocab_size)
        x = self.pos_embedding(x, pos_phase=pos_phase)
        # (batch_size, target_seq_len, d_model)
        x = self.dropout(x)

        for i in range(self.num_layers):
            x = self.dec_layers[i](x=x, training=training)

        # self.last_attn_scores = self.dec_layers[-1].last_attn_scores

        # The shape of x is (batch_size, target_seq_len, d_model).
        return x

# Entire transformer model


class Transformer(tf.keras.Model):
    def __init__(self, *, num_layers, d_model, num_heads, dff,
                 input_vocab_size, target_vocab_size, n_qubits, dropout_rate=0.0, init_bias="zeros", **kwargs):
        super().__init__()

        self.decoder = Decoder(num_layers=num_layers, d_model=d_model,
                               num_heads=num_heads, dff=dff,
                               vocab_size=target_vocab_size,
                               n_qubits=n_qubits,
                               dropout_rate=dropout_rate)

        bi = tf.constant_initializer(init_bias)
        self.final_layer = tf.keras.layers.Dense(
            target_vocab_size, bias_initializer=bi, kernel_initializer="zeros")

        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff

        self.input_vocab_size = input_vocab_size
        self.target_vocab_size = target_vocab_size
        self.n_qubits = n_qubits

        self.dropout_rate = dropout_rate
        self.init_bias = init_bias

    def get_config(self):
        config = {
            "num_layers": self.num_layers,
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "dff": self.dff,
            "input_vocab_size": self.input_vocab_size,
            "target_vocab_size": self.target_vocab_size,
            "n_qubits": self.n_qubits,
            "dropout_rate": self.dropout_rate,
            "init_bias": self.init_bias,
        }

        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, inputs, training, pos_phase=0):
        # To use a Keras model with `.fit` you must pass all your inputs in the
        # first argument.
        # (batch_size, target_len, target_vocab_size)
        x = self.decoder(x=inputs, training=training, pos_phase=pos_phase)

        # Final linear layer output.
        # (batch_size, target_len, target_vocab_size)
        logits = self.final_layer(x)

        try:
            # Drop the keras mask, so it doesn't scale the losses/metrics.
            # b/250038731
            del logits._keras_mask
        except AttributeError:
            pass

        # Return the final output and the attention weights.
        return logits


if __name__ == "__main__":
    config = read_config()

    print("Building model...")
    transformer = Transformer(
        num_layers=config["NUM_LAYERS"],
        d_model=config["D_MODEL"],
        num_heads=config["NUM_HEADS"],
        dff=config["D_FFN"],
        input_vocab_size=config["TARGET_VOCAB_SIZE"],  # same as target size
        target_vocab_size=config["TARGET_VOCAB_SIZE"],
        n_qubits=config["N_QUBITS"],
        dropout_rate=config["DROPOUT_RATE"])

    input_tensor = tf.constant([[0], [1]], dtype=tf.float16)
    print(input_tensor)
    # input_tensor = tf.zeros([10, 1])

    output = transformer(inputs=input_tensor, training=False)

    print(output)
    # print(output[:, -1, :])

    # print(transformer.summary())
