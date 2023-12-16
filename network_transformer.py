from typing import Any
from functools import partial
import time
import dataclasses
import itertools

from tqdm import tqdm

import numpy as np
import jax
from jax import random, numpy as jnp
from flax import linen as nn
from flax.training import train_state
import orbax.checkpoint
from flax.training import orbax_utils
import optax

import wandb

from buffer import load_batch, Batch


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
        t_embed = nn.Embed(200, self.embed_dim)(tokens[..., 4])

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


class MultiHeadLinearAttentionWithCache(nn.Module):
    num_heads: int
    embed_dim: int

    @nn.compact
    def __call__(self, x, s, z):
        head_dim = self.embed_dim // self.num_heads

        v = nn.Dense(features=self.embed_dim)(x)  # [Head * Dim]
        q = nn.Dense(features=self.embed_dim)(x)  # [Head * Dim]
        k = nn.Dense(features=self.embed_dim)(x)  # [Head * Dim]

        v = v.reshape(self.num_heads, head_dim)  # [Batch, SeqLen, Head, Dim]
        q = q.reshape(self.num_heads, head_dim)  # [Batch, SeqLen, Head, Dim]
        k = k.reshape(self.num_heads, head_dim)  # [Batch, SeqLen, Head, Dim]

        q = nn.elu(q) + 1
        k = nn.elu(k) + 1

        s += jnp.einsum('hd,hm->hdm', k, v)
        z += k

        qs = jnp.einsum('hd,hdm->hm', q, s)
        qz = jnp.einsum('hd,hd->h', q, z).reshape(self.num_heads, 1)

        values = qs / (qz + 1E-6)

        values = values.reshape(self.num_heads * head_dim)  # [Head × Dim (=EmbedDim)]
        out = nn.Dense(self.embed_dim)(values)

        return out, s, z


class MultiHeadLinearAttention(nn.Module):
    num_heads: int
    embed_dim: int

    def test(self, i, v, q, k, s, z):
        s += jnp.einsum('bhd,bhm->bhdm', k[:, i], v[:, i])
        z += k[:, i]

        q_i = q[:, i]

        qs = jnp.einsum('bhd,bhdm->bhm', q_i, s)
        qz = jnp.einsum('bhd,bhd->bh', q_i, z).reshape(-1, self.num_heads, 1)

        return (s, z), qs / (qz + 1E-6)

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

        q = nn.elu(q) + 1
        k = nn.elu(k) + 1

        s = jnp.zeros((x.shape[0], self.num_heads, head_dim, head_dim))
        z = jnp.zeros((x.shape[0], self.num_heads, head_dim))

        (s, z), values = jax.lax.scan(lambda t, i: self.test(i, v, q, k, *t),
                                      init=(s, z),
                                      xs=jnp.arange(seq_len))

        values = values.transpose(1, 0, 2, 3)

        values = values.reshape(-1, seq_len, self.num_heads * head_dim)  # [Batch, SeqLen, Head × Dim (=EmbedDim)]
        out = nn.Dense(self.embed_dim)(values)

        return out


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
        # x: [EmbedDim]
        # v: [SeqLen k, Head, Dim]
        # k: [SeqLen k, Head, Dim]
        head_dim = self.embed_dim // self.num_heads

        # [1, Head, Dim]
        v_i = nn.Dense(features=self.embed_dim)(x).reshape(1, self.num_heads, head_dim)
        q_i = nn.Dense(features=self.embed_dim)(x).reshape(self.num_heads, head_dim)
        k_i = nn.Dense(features=self.embed_dim)(x).reshape(1, self.num_heads, head_dim)

        # [SeqLen k, Head, Dim]
        v = jnp.concatenate((v[1:], v_i), axis=0)
        k = jnp.concatenate((k[1:], k_i), axis=0)

        mask = jnp.any(v != 0, axis=(1, 2))

        # [Head, SeqLen k]
        attention = jnp.einsum('hd,khd->hk', q_i, k) / jnp.sqrt(head_dim)
        attention = jnp.where(mask, attention, -jnp.inf)
        attention = nn.softmax(attention, axis=-1)

        # [Head, Dim]
        values = jnp.einsum('hk,khd->hd', attention, v)
        # [EmbedDim]
        values = values.reshape(self.embed_dim)
        # [EmbedDim]
        out_i = nn.Dense(self.embed_dim)(values)

        return out_i, v, k


class FeedForward(nn.Module):
    embed_dim: int
    intermediate_size: int = 128

    @nn.compact
    def __call__(self, x, eval):
        x = nn.Dense(features=self.embed_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.embed_dim)(x)
        x = nn.Dropout(0.1, deterministic=eval)(x)
        return x


class TransformerBlock(nn.Module):
    num_heads: int
    embed_dim: int
    is_linear: bool

    def setup(self):
        if self.is_linear:
            self.attention = MultiHeadLinearAttention(self.num_heads, self.embed_dim)
        else:
            self.attention = MultiHeadAttention(self.num_heads, self.embed_dim)

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
    is_linear: bool

    def setup(self):
        if self.is_linear:
            self.attention = MultiHeadLinearAttentionWithCache(
                num_heads=self.num_heads,
                embed_dim=self.embed_dim)
        else:
            self.attention = MultiHeadAttentionWithCache(
                num_heads=self.num_heads,
                embed_dim=self.embed_dim)

        self.feed_forward = FeedForward(embed_dim=self.embed_dim)

    @nn.compact
    def __call__(self, x, cache1, cache2, eval):
        out, cache1, cache2 = self.attention(x, cache1, cache2)

        x = x + out
        x = nn.LayerNorm()(x)
        x = x + self.feed_forward(x, eval)
        x = nn.LayerNorm()(x)

        return x, cache1, cache2


NUM_ACTIONS = 32
MAX_LENGTH = 200


class StateEncoder(nn.Module):
    embed_dim: int
    n_blocks: int
    n_filters: int

    @nn.compact
    def __call__(self, x):
        # x: [..., 6, 6, 16]
        # return: [..., EmbedDim]
        org_shape = x.shape
        x = x.reshape(-1, *org_shape[-3:])

        for _ in range(self.n_blocks):
            x = nn.Conv(self.n_filters, kernel_size=(3, 3))(x)
            x = nn.relu(x)

        x = x.reshape(*org_shape[:-3], -1)
        x = nn.Dense(self.embed_dim)(x)
        x = nn.tanh(x)

        return x


class TransformerDecoder(nn.Module):
    num_heads: int
    embed_dim: int
    num_hidden_layers: int
    is_linear_attention: bool

    has_state_encoder: bool = False
    state_encoder_n_blocks: int = 1
    state_encoder_n_filters: int = 64

    max_n_ply: int = 200

    def setup(self):
        self.embeddings = Embeddings(self.embed_dim, max_n_ply=self.max_n_ply)

        self.layers = [TransformerBlock(self.num_heads, self.embed_dim, self.is_linear_attention)
                       for _ in range(self.num_hidden_layers)]

        if self.has_state_encoder:
            self.state_encoder = StateEncoder(self.embed_dim ** 2 / self.num_heads,
                                              self.state_encoder_n_blocks,
                                              self.state_encoder_n_filters)

    @nn.compact
    def __call__(self, x, states=None, eval=True):
        # [Batch, 1, SeqLen, SeqLen]
        mask = nn.make_causal_mask(jnp.zeros((x.shape[0], x.shape[1])), dtype=bool)
        x = self.embeddings(x, eval)

        if self.has_state_encoder:
            x += self.state_encoder(states)
            x = nn.LayerNorm()(x)

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
    is_linear_attention: bool = False

    has_state_encoder: bool = False
    state_encoder_n_blocks: int = 1
    state_encoder_n_filters: int = 64

    max_n_ply: int = 200

    def setup(self):
        self.embeddings = Embeddings(self.embed_dim, max_n_ply=self.max_n_ply)
        self.layers = [TransformerBlockWithCache(self.num_heads, self.embed_dim, self.is_linear_attention)
                       for _ in range(self.num_hidden_layers)]

    def create_cache(self, seq_len):
        head_dim = self.embed_dim // self.num_heads

        # [Batch, Layer, SeqLen, Head, HeadDim]
        v = jnp.zeros((self.num_hidden_layers, seq_len, self.num_heads, head_dim))
        k = jnp.zeros((self.num_hidden_layers, seq_len, self.num_heads, head_dim))

        return v, k

    def create_linear_cache(self):
        head_dim = self.embed_dim // self.num_heads

        s = jnp.zeros((self.num_hidden_layers, self.num_heads, head_dim, head_dim))
        z = jnp.zeros((self.num_hidden_layers, self.num_heads, head_dim))

        return s, z

    @nn.compact
    def __call__(self, x, cache1, cache2, eval=True):
        x = self.embeddings(x, eval)

        # if not self.is_linear_attention:
        if False:
            next_cache1, next_cache2 = self.create_cache(cache1.shape[1] + 1)
        else:
            next_cache1, next_cache2 = cache1, cache2

        i = 0
        for layer in self.layers:
            x, cache1_i, cache2_i = layer(x, cache1[i], cache2[i], eval=eval)

            next_cache1 = next_cache1.at[i].set(cache1_i)
            next_cache2 = next_cache2.at[i].set(cache2_i)

            i += 1

        x = nn.Dropout(0.1, deterministic=eval)(x)

        logits_pi = nn.Dense(features=NUM_ACTIONS)(x)
        logits_v = nn.Dense(features=7)(x)
        logits_color = nn.Dense(features=8)(x)

        return logits_pi, logits_v, logits_color, next_cache1, next_cache2


class TrainState(train_state.TrainState):
    epoch: int
    dropout_rng: Any


@partial(jax.jit, static_argnames=['eval'])
def loss_fn(params, state, x, y_pi, y_v, y_color, dropout_rng, eval):
    pi, v, color = state.apply_fn({'params': params}, x, eval=eval,
                                  rngs={'dropout': dropout_rng})

    mask = jnp.any(x != 0, axis=2)

    # [Batch, SeqLen, 32]
    y_pi = y_pi.reshape(-1, x.shape[1])
    y_v = y_v.reshape(-1, 1)
    y_color = y_color.reshape(-1, 1, 8)

    loss_pi = optax.softmax_cross_entropy_with_integer_labels(pi, y_pi)
    loss_v = optax.softmax_cross_entropy_with_integer_labels(v, y_v)
    loss_color = optax.sigmoid_binary_cross_entropy(color, y_color).mean(axis=2)

    loss_pi = jnp.average(loss_pi, weights=mask)
    loss_v = jnp.average(loss_v, weights=mask)
    loss_color = jnp.average(loss_color, weights=mask)

    loss = loss_pi + loss_v + loss_color

    info = jnp.array([loss_pi, loss_v, loss_color])

    return loss, info


@partial(jax.jit, static_argnames=['eval'])
def train_step(state: TrainState, x, y_pi, y_v, y_color, eval):
    if not eval:
        new_dropout_rng, dropout_rng = random.split(state.dropout_rng)
        (loss, info), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            state.params, state, x, y_pi, y_v, y_color, dropout_rng, eval)

        new_state = state.apply_gradients(grads=grads, dropout_rng=new_dropout_rng)
    else:
        loss, info = loss_fn(
            state.params, state, x, y_pi, y_v, y_color, random.PRNGKey(0), eval)
        new_state = state

    return new_state, loss, info


def train_epoch(state, data_batched, eval):
    loss_history, info_history = [], []
    for x, mask, y_pi, y_v, y_color in tqdm(zip(*data_batched), total=len(data_batched[0])):
        state, loss, info = train_step(state, x, mask, y_pi, y_v, y_color, eval)
        loss_history.append(jax.device_get(loss))
        info_history.append(jax.device_get(info))
    return state, jnp.mean(jnp.array(loss_history)), jnp.mean(jnp.array(info_history), axis=0)


def create_batches(data, batch_size):
    n = len(data) // batch_size
    data_batched = data[:n*batch_size].reshape(n, batch_size, *data.shape[1:])
    return data_batched


def fit_as_rl_simulation(state: TrainState,
                         model: TransformerDecoder,
                         checkpoint_manager,
                         batch: Batch,
                         buffer_size=400000,
                         batch_size=256,
                         num_batches=8,
                         update_period=200):

    num_iter = (batch.tokens.shape[0] - buffer_size) // update_period
    for step in range(num_iter):
        indices = np.arange(buffer_size) + step * update_period

        info = np.zeros((num_batches, 3))
        loss = 0

        for i in range(num_batches):
            indices_batch = np.random.choice(indices, size=batch_size)
            batch_i = batch.create_batch_from_indices(indices_batch)
            state, loss_i, info_i = train_step(state, *batch_i.astuple(), eval=False)

            loss += loss_i
            info[i] = info_i

        info = info.mean(axis=0)
        loss /= num_batches

        state = state.replace(epoch=state.epoch + 1)

        if step % 1000 == 0:
            save_checkpoint(state, model, checkpoint_manager)

        print(step, loss, info)


def fit(state, model, checkpoint_manager, train_data, test_data, epochs, batch_size, log_wandb):
    train_data_batched = [create_batches(data, batch_size) for data in train_data]
    test_data_batched = [create_batches(data, batch_size) for data in test_data]

    for epoch in range(state.epoch + 1, state.epoch + 1 + epochs):
        start = time.perf_counter()

        state, loss_train, info_train = train_epoch(state, train_data_batched, eval=False)
        _, loss_test, info_test = train_epoch(state, test_data_batched, eval=True)

        elapsed_time = time.perf_counter() - start

        print(f'Epoch: {epoch}, ', end='')
        print(f'Loss: ({loss_train:.3f}, {loss_test:.3f}), ', end='')
        print(f'P: ({info_train[0]:.3f}, {info_test[0]:.3f}), ', end='')
        print(f'V: ({info_train[1]:.3f}, {info_test[1]:.3f}), ', end='')
        print(f'C: ({info_train[2]:.3f}, {info_test[2]:.3f})')

        log_dict = {
            'epoch': epoch,
            'elapsed time': elapsed_time,
            'train/loss': loss_train,
            'train/loss policy': info_train[0],
            'train/loss value': info_train[1],
            'train/loss color': info_train[2],
            'test/loss': loss_test,
            'test/loss policy': info_test[0],
            'test/loss value': info_test[1],
            'test/loss color': info_test[2],
        }
        if log_wandb:
            wandb.log(log_dict)

        state = state.replace(epoch=state.epoch + 1)
        save_checkpoint(state, model, checkpoint_manager)

    return state


def main_train_as_rl_simulation(batch: Batch):
    h = 4
    d = 256
    n = 4
    model = TransformerDecoder(num_heads=h, embed_dim=d, num_hidden_layers=n, is_linear_attention=False, max_n_ply=220)

    checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    checkpoint_manager = orbax.checkpoint.CheckpointManager('./checkpoints/4_256_4_run-2', checkpointer)

    ckpt = checkpoint_manager.restore(8)

    key, key1, key2 = random.split(random.PRNGKey(0), 3)

    state = TrainState.create(
        apply_fn=model.apply,
        params=ckpt['state']['params'],
        tx=optax.adam(learning_rate=0.0005),
        dropout_rng=ckpt['state']['dropout_rng'],
        epoch=ckpt['state']['epoch'])

    ckpt_dir = f'./checkpoints/{h}_{d}_{n}_run-2-e'

    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    options = orbax.checkpoint.CheckpointManagerOptions(create=True)
    checkpoint_manager = orbax.checkpoint.CheckpointManager(ckpt_dir, orbax_checkpointer, options)

    save_checkpoint(state, model, checkpoint_manager)

    fit_as_rl_simulation(state, model, checkpoint_manager, batch)

    save_checkpoint(state, model, checkpoint_manager)


def main_train(data, log_wandb=False):
    train_n = int(len(data[0]) * 0.8)

    train_data = [d[:train_n] for d in data]
    test_data = [d[train_n:] for d in data]

    key, key1, key2 = random.split(random.PRNGKey(0), 3)

    heads = 4,
    dims = 256,
    num_layers = 4,

    for h, d, n in itertools.product(heads, dims, num_layers):
        if log_wandb:
            name = f'h={h}, d={d}, n={n}'
            run_config = {
                'num heads': h,
                'embed dim': d,
                'num layers': n,
            }
            run = wandb.init(project='network benchmark', config=run_config, name=name)

        model = TransformerDecoder(num_heads=h, embed_dim=d, num_hidden_layers=n, is_linear_attention=False)

        variables = model.init(key1, data[0][:1])
        state = TrainState.create(
            apply_fn=model.apply,
            params=variables['params'],
            tx=optax.adam(learning_rate=0.0005),
            dropout_rng=key2,
            epoch=0)

        ckpt_dir = f'./checkpoints/{h}_{d}_{n}_run-2'

        orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        options = orbax.checkpoint.CheckpointManagerOptions(create=True)
        checkpoint_manager = orbax.checkpoint.CheckpointManager(ckpt_dir, orbax_checkpointer, options)

        save_checkpoint(state, model, checkpoint_manager)

        state = fit(state, model, checkpoint_manager,
                    train_data=train_data,
                    test_data=test_data,
                    epochs=8, batch_size=64,
                    log_wandb=log_wandb)

        if log_wandb:
            run.finish()


def create_ckpt(state: TrainState, model: TransformerDecoder):
    return {'state': state, 'model': dataclasses.asdict(model)}


def save_checkpoint(state: TrainState,
                    model: TransformerDecoder,
                    checkpoint_manager: orbax.checkpoint.CheckpointManager):
    ckpt = create_ckpt(state, model)
    save_ckpt(state.epoch, ckpt, checkpoint_manager)


def save_ckpt(step, ckpt, checkpoint_manager: orbax.checkpoint.CheckpointManager):
    save_args = orbax_utils.save_args_from_target(ckpt)
    checkpoint_manager.save(step, ckpt, save_kwargs={'save_args': save_args})


@partial(jax.jit, device=jax.devices("cpu")[0])
def predict(state, x, cache1, cache2):
    return state.apply_fn({'params': state.params}, x, cache1, cache2)


def main_test_performance(data):
    is_linear = False

    ckpt_dir = './checkpoints/'

    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    checkpoint_manager = orbax.checkpoint.CheckpointManager(ckpt_dir, orbax_checkpointer)

    ckpt = checkpoint_manager.restore(checkpoint_manager.latest_step())

    model = TransformerDecoderWithCache(**ckpt['model'])

    state = TrainState.create(
        apply_fn=model.apply,
        params=ckpt['state']['params'],
        tx=optax.adam(learning_rate=0.0005),
        dropout_rng=ckpt['state']['dropout_rng'],
        epoch=0)

    def test(data_index):
        if is_linear:
            cache1, cache2 = model.create_linear_cache()
        else:
            cache1, cache2 = model.create_cache(200)

        v_history = np.zeros(200)

        for i in range(100):
            pi, v, color, cache1, cache2 = predict(state, data[0][data_index, i], cache1, cache2)
            v_history[i] = jnp.sum(nn.softmax(v) * jnp.array([-1, -1, -1, 0, 1, 1, 1]))

        return v_history

    start = time.perf_counter()

    v_mean_history = np.zeros(100)

    for i in range(v_mean_history.shape[0]):
        v_history = test(i)
        v_mean_history[i] = v_history.mean()

    print(v_mean_history)

    t = time.perf_counter() - start
    print(t / v_mean_history.shape[0])


def main():
    batch = load_batch(['replay_buffer/run-2-e.npz'], shuffle=False)
    print(np.mean(np.sum(batch.mask, axis=1)))
    data = batch.astuple()

    print(data[0].shape)

    # main_test_performance(data)
    main_train_as_rl_simulation(batch)
    # main_train(data)


if __name__ == "__main__":
    main()
