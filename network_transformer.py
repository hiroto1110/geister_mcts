from typing import Any
import time
from functools import partial

import numpy as np
import jax
from jax import random, numpy as jnp
from flax import linen as nn
from flax.training import train_state, checkpoints
import optax

import matplotlib.pyplot as plt


class Embeddings(nn.Module):
    embed_dim: int
    piece_type: int = 5
    n_pieces: int = 16
    board_size: int = 7
    max_n_ply: int = 200

    @nn.compact
    def __call__(self, tokens, eval):
        type_embed = nn.Embed(self.piece_type, self.embed_dim)(tokens[..., 0])
        id_embed = nn.Embed(self.n_pieces, self.embed_dim)(tokens[..., 1])
        x_embed = nn.Embed(self.board_size, self.embed_dim)(tokens[..., 2])
        y_embed = nn.Embed(self.board_size, self.embed_dim)(tokens[..., 3])
        t_embed = nn.Embed(self.max_n_ply, self.embed_dim)(tokens[..., 4])

        embeddings = type_embed + id_embed + x_embed + y_embed + t_embed
        embeddings = nn.LayerNorm(epsilon=1e-12)(embeddings)
        embeddings = nn.Dropout(0.5, deterministic=eval)(embeddings)

        return embeddings


class RelativePostionalEncoding(nn.Module):
    num_heads: int
    embed_dim: int
    max_x: int

    @nn.compact
    def __call__(self, q, x):
        seq_len = x.shape[1]
        head_dim = self.embed_dim // self.num_heads

        # [Batch, SeqLen, SeqLen]
        rel_x = self.max_x + x.reshape(-1, 1, seq_len) - x.reshape(-1, seq_len, 1)
        # [Batch, SeqLen, SeqLen, Dim]
        rel_x = nn.Embed(self.max_x * 2 + 1, self.embed_dim)(rel_x)
        rel_x = rel_x.reshape(-1, seq_len, seq_len, self.num_heads, head_dim)
        # [Batch, Head, SeqLen, SeqLen]
        return jnp.einsum('...qhd,...qkhd->...hqk', q, rel_x)


class MultiHeadAttention(nn.Module):
    num_heads: int
    embed_dim: int

    @nn.compact
    def __call__(self, x, mask):
        seq_len = x.shape[1]
        head_dim = self.embed_dim // self.num_heads

        v = nn.Dense(features=self.embed_dim)(x)  # [Batch, SeqLen, Head * Dim]
        q = nn.Dense(features=self.embed_dim)(x)  # [Batch, SeqLen, Head * Dim]
        k = nn.Dense(features=self.embed_dim)(x)  # [Batch, SeqLen, Head * Dim]

        v = v.reshape(-1, seq_len, self.num_heads, head_dim)  # [Batch, SeqLen, Head, Dim]
        q = q.reshape(-1, seq_len, self.num_heads, head_dim)  # [Batch, SeqLen, Head, Dim]
        k = k.reshape(-1, seq_len, self.num_heads, head_dim)  # [Batch, SeqLen, Head, Dim]

        # [Batch, Head, SeqLen, SeqLen]
        attention = (jnp.einsum('...qhd,...khd->...hqk', q, k) / jnp.sqrt(head_dim))

        attention = jnp.where(mask, attention, -jnp.inf)
        attention = nn.softmax(attention, axis=-1)

        values = jnp.einsum('...hqk,...khd->...qhd', attention, v)  # [Batch, SeqLen, Head, Dim]
        values = values.reshape(-1, seq_len, self.num_heads * head_dim)  # [Batch, SeqLen, Head × Dim (=EmbedDim)]
        out = nn.Dense(self.embed_dim)(values)

        return out


class MultiHeadAttentionWithCache(nn.Module):
    num_heads: int
    embed_dim: int

    @nn.compact
    def __call__(self, x, v, k):
        # x: [Batch, SeqLen q, EmbedDim]
        # v: [Batch, SeqLen k, Head, Dim]
        # k: [Batch, SeqLen k, Head, Dim]
        head_dim = self.embed_dim // self.num_heads
        seq_len_q = x.shape[1]

        # [Batch, 1, Head, Dim]
        v_i = nn.Dense(features=self.embed_dim)(x).reshape(-1, seq_len_q, self.num_heads, head_dim)
        q_i = nn.Dense(features=self.embed_dim)(x).reshape(-1, seq_len_q, self.num_heads, head_dim)
        k_i = nn.Dense(features=self.embed_dim)(x).reshape(-1, seq_len_q, self.num_heads, head_dim)

        # [Batch, SeqLen k + SeqLen q, Head, Dim]
        v = jnp.concatenate((v, v_i), axis=1)
        k = jnp.concatenate((k, k_i), axis=1)

        # [Batch, Head, SeqLen q, SeqLen k + SeqLen q]
        attention = jnp.einsum('...qhd,...khd->...hqk', q_i, k) / jnp.sqrt(head_dim)
        attention = nn.softmax(attention, axis=-1)

        # [Batch, SeqLen q, Head, Dim]
        values = jnp.einsum('...hqk,...khd->...qhd', attention, v)
        # [Batch, SeqLen q, EmbedDim]
        values = values.reshape(-1, seq_len_q, self.embed_dim)
        # [Batch, SeqLen q, EmbedDim]
        out_i = nn.Dense(self.embed_dim)(values)

        return v, k, out_i


class FeedForward(nn.Module):
    embed_dim: int
    intermediate_size: int = 2048

    @nn.compact
    def __call__(self, x, eval):
        x = nn.Dense(features=self.intermediate_size)(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.embed_dim)(x)
        x = nn.Dropout(0.1, deterministic=eval)(x)
        return x


class TransformerBlock(nn.Module):
    num_heads: int
    embed_dim: int

    def setup(self):
        self.attention = MultiHeadAttention(
            num_heads=self.num_heads,
            embed_dim=self.embed_dim)

        self.feed_forward = FeedForward(embed_dim=self.embed_dim)

    @nn.compact
    def __call__(self, x, attention_mask, eval):
        out = self.attention(x, attention_mask)

        x = x + out
        x = nn.LayerNorm()(x)
        x = x + self.feed_forward(x, eval)
        x = nn.LayerNorm()(x)
        return x


class TransformerBlockWithCache(nn.Module):
    num_heads: int
    embed_dim: int

    def setup(self):
        self.attention = MultiHeadAttentionWithCache(
            num_heads=self.num_heads,
            embed_dim=self.embed_dim)

        self.feed_forward = FeedForward(embed_dim=self.embed_dim)

    @nn.compact
    def __call__(self, x, v, k, eval):
        v, k, out = self.attention(x, v, k)

        x = x + out
        x = nn.LayerNorm()(x)
        x = x + self.feed_forward(x, eval)
        x = nn.LayerNorm()(x)

        return x, v, k


NUM_ACTIONS = 32
MAX_LENGTH = 200


class TransformerDecoder(nn.Module):
    num_heads: int
    embed_dim: int
    num_hidden_layers: int

    def setup(self):
        self.embeddings = Embeddings(self.embed_dim)

        self.layers = [TransformerBlock(num_heads=self.num_heads, embed_dim=self.embed_dim)
                       for _ in range(self.num_hidden_layers)]

    @nn.compact
    def __call__(self, x, eval=True):
        # [Batch, 1, SeqLen, SeqLen]
        mask = nn.make_causal_mask(jnp.zeros((x.shape[0], x.shape[1])), dtype=bool)
        x = self.embeddings(x, eval)

        for i in range(self.num_hidden_layers):
            x = self.layers[i](x, mask, eval=eval)
        x = nn.Dropout(0.1, deterministic=eval)(x)

        pi = nn.Dense(features=NUM_ACTIONS)(x)
        v = nn.Dense(features=7)(x)
        color = nn.Dense(features=8)(x)

        return pi, v, color  # [Batch, SeqLen, ...]


class TransformerDecoderWithCache(nn.Module):
    num_heads: int
    embed_dim: int
    num_hidden_layers: int

    def setup(self):
        self.embeddings = Embeddings(self.embed_dim)
        self.layers = [TransformerBlockWithCache(num_heads=self.num_heads, embed_dim=self.embed_dim)
                       for _ in range(self.num_hidden_layers)]

    def create_cache(self, batch_size, seq_len):
        head_dim = self.embed_dim // self.num_heads

        # [Batch, Layer, SeqLen, Head, HeadDim]
        v = jnp.zeros((self.num_hidden_layers, batch_size, seq_len, self.num_heads, head_dim))
        k = jnp.zeros((self.num_hidden_layers, batch_size, seq_len, self.num_heads, head_dim))

        return v, k

    @nn.compact
    def __call__(self, x, v, k, eval=True):
        x = self.embeddings(x, eval)

        next_v, next_k = self.create_cache(x.shape[0], v.shape[2] + x.shape[1])

        i = 0
        for layer in self.layers:
            x, v_i, k_i = layer(x, v[i], k[i], eval=eval)

            next_v = next_v.at[i].set(v_i)
            next_k = next_k.at[i].set(k_i)

            i += 1

        x = nn.Dropout(0.1, deterministic=eval)(x)

        logits_pi = nn.Dense(features=NUM_ACTIONS)(x)
        logits_v = nn.Dense(features=7)(x)
        logits_color = nn.Dense(features=8)(x)

        return logits_pi, logits_v, logits_color, next_v, next_k


@partial(jax.jit, static_argnames=['eval'])
def loss_fn(params, state, x, mask, y_pi, y_v, y_color, dropout_rng, eval):
    pi, v, color = state.apply_fn({'params': params}, x, eval=eval,
                                  rngs={'dropout': dropout_rng})

    # [Batch, SeqLen, 144]
    y_pi = jnp.clip(y_pi.reshape((-1, x.shape[1])), 0, 31)
    y_v = jnp.clip(y_v.reshape((-1, 1)), 0, 6)
    # y_v = jnp.clip(y_v.reshape((-1, 1, 1)), -1, 1)
    y_color = y_color.reshape((-1, 1, 8))

    loss_pi = mask * optax.softmax_cross_entropy_with_integer_labels(pi, y_pi)
    # loss_pi = optax.softmax_cross_entropy(pi, y_pi).mean(axis=0)
    loss_v = mask * optax.softmax_cross_entropy_with_integer_labels(v, y_v)
    # loss_v = jnp.mean((v - y_v) ** 2, axis=(0, 2))
    loss_color = mask * optax.sigmoid_binary_cross_entropy(color, y_color).mean(axis=2)

    loss = jnp.mean(0.5 * loss_pi + loss_v + loss_color)

    loss_pi = loss_pi.mean(axis=0)
    loss_v = loss_v.mean(axis=0)
    loss_color = loss_color.mean(axis=0)

    acc_piece = jnp.mean((color > 0) == y_color, axis=(0, 2))

    info = jnp.stack([loss_pi, loss_v, loss_color, acc_piece], axis=0)

    return loss, info


@partial(jax.jit, static_argnames=['eval'])
def train_step(state, x, mask, y_pi, y_v, y_color, eval):
    if not eval:
        new_dropout_rng, dropout_rng = random.split(state.dropout_rng)
        (loss, info), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            state.params, state, x, mask, y_pi, y_v, y_color, dropout_rng, eval)

        new_state = state.apply_gradients(grads=grads, dropout_rng=new_dropout_rng)
    else:
        loss, info = loss_fn(
            state.params, state, x, mask, y_pi, y_v, y_color, random.PRNGKey(0), eval)
        new_state = state

    return new_state, loss, info


def train_epoch(state, data_batched, eval):
    loss_history, info_history = [], []
    for x, y_pi, y_v, y_color in zip(*data_batched):
        state, loss, info = train_step(state, x, 1, y_pi, y_v, y_color, eval)
        loss_history.append(jax.device_get(loss))
        info_history.append(jax.device_get(info))
    return state, jnp.mean(jnp.array(loss_history)), jnp.mean(jnp.array(info_history), axis=(0, 2))


def create_batches(data, batch_size):
    n = len(data) // batch_size
    data_batched = data[:n*batch_size].reshape(n, batch_size, *data.shape[1:])
    return data_batched


def fit(state, ckpt_dir, prefix, train_data, test_data, epochs, batch_size):
    # state = checkpoints.restore_checkpoint(
    #    ckpt_dir=ckpt_dir, prefix=prefix, target=state)

    train_data_batched = [create_batches(data, batch_size) for data in train_data]
    test_data_batched = [create_batches(data, batch_size) for data in test_data]

    loss_history_train, acc_history_train = [], []
    loss_history_test, acc_history_test = [], []

    for epoch in range(state.epoch + 1, state.epoch + 1 + epochs):
        # Training
        state, loss_train, info_train = train_epoch(state, train_data_batched, eval=False)
        loss_history_train.append(loss_train)
        acc_history_train.append(info_train)

        # Evaluation
        _, loss_test, info_test = train_epoch(state, test_data_batched, eval=True)
        loss_history_test.append(loss_test)
        acc_history_test.append(info_test)

        print(f'Epoch: {epoch}, ', end='')
        print(f'Loss: ({loss_train:.3f}, {loss_test:.3f}), ', end='')
        print(f'Pi: ({info_train[0]:.3f}, {info_test[0]:.3f}), ', end='')
        print(f'V: ({info_train[1]:.3f}, {info_test[1]:.3f}), ', end='')
        print(f'Color: ({info_train[2]:.3f}, {info_test[2]:.3f}), ', end='')
        print(f'Acc Color: ({info_train[3]:.3f}, {info_test[3]:.3f})')

        state = state.replace(epoch=state.epoch + 1)
        checkpoints.save_checkpoint(
            ckpt_dir=ckpt_dir, prefix=prefix,
            target=state, step=state.epoch, overwrite=True, keep=5)

    history = {'loss_train': loss_history_train,
               'acc_train': acc_history_train,
               'loss_test': loss_history_test,
               'acc_test': acc_history_test}

    return state, history


class TrainState(train_state.TrainState):
    epoch: int
    dropout_rng: Any


def main_train(model, state, data):
    train_n = int(len(data[0]) * 0.8)

    train_data = [d[:train_n] for d in data]
    test_data = [d[train_n:] for d in data]

    ckpt_dir = './checkpoints/'
    prefix = 'geister_'

    checkpoints.save_checkpoint(
        ckpt_dir=ckpt_dir, prefix=prefix,
        target=state, step=state.epoch, overwrite=True)

    start = time.perf_counter()

    state, history = fit(state, ckpt_dir, prefix,
                         train_data=train_data,
                         test_data=test_data,
                         epochs=8, batch_size=32)

    end = time.perf_counter()
    print(end - start)


def main_test(model, state, data):
    start = time.perf_counter()

    n = 128
    pi, v, c, attention = model.apply({'params': state.params}, data[0][:n], eval=True)

    end = time.perf_counter()
    print(end - start)

    y_pi = data[1][:n].reshape((-1, MAX_LENGTH))
    y_v = jnp.tile(data[2][:n], MAX_LENGTH).reshape((-1, MAX_LENGTH, 1))
    y_c = jnp.tile(data[3][:n], MAX_LENGTH).reshape((-1, MAX_LENGTH, 8))

    loss_policy = optax.softmax_cross_entropy_with_integer_labels(pi, y_pi).mean(axis=(0,))
    loss_reward = optax.squared_error(v, y_v).mean(axis=(0, 2))
    loss_piece = optax.sigmoid_binary_cross_entropy(c, y_c).mean(axis=(0, 2))

    acc_piece = jnp.mean((c > 0) == y_c, axis=(0, 2))

    plt.plot(loss_policy)
    plt.savefig("fig_loss_policy")
    plt.clf()

    plt.plot(loss_reward)
    plt.savefig("fig_loss_reward")
    plt.clf()

    plt.plot(loss_piece)
    plt.savefig("fig_loss_piece")
    plt.clf()

    plt.plot(acc_piece)
    plt.savefig("fig_acc_piece")


def main():
    if False:
        tokens_buffer = np.load('replay_buffer/tokens.npy')
        policy_buffer = np.load('replay_buffer/policy.npy')
        reward_buffer = np.load('replay_buffer/reward.npy') + 3
        pieces_buffer = np.load('replay_buffer/pieces.npy')
        data = tokens_buffer, policy_buffer, reward_buffer, pieces_buffer
    else:
        data = [jnp.load(f"data_{i}.npy") for i in range(4)]

    print(data[0].shape)

    model = TransformerDecoder(num_heads=8, embed_dim=128, num_hidden_layers=3)

    key, key1, key2 = random.split(random.PRNGKey(0), 3)

    variables = model.init(key1, data[0][:1])
    state = TrainState.create(
        apply_fn=model.apply,
        params=variables['params'],
        tx=optax.adam(learning_rate=0.00005),
        dropout_rng=key2,
        epoch=0)

    # ckpt_dir = './checkpoints/'
    # prefix = 'geister_'

    # print(variables['params'].keys())

    # state = checkpoints.restore_checkpoint(ckpt_dir=ckpt_dir, prefix=prefix, target=state)

    main_train(model, state, data)
    # main_test(model, state, data)


if __name__ == "__main__":
    main()
