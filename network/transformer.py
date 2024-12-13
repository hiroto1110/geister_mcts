from __future__ import annotations

from functools import partial
import dataclasses
import serde

import jax
import optax
from jax import random, numpy as jnp
from flax import linen as nn

from network.train_state import TrainStateBase
from batch import FORMAT_X7ARC


@serde.serde
@dataclasses.dataclass(frozen=True)
class TransformerConfig:
    num_heads: int
    embed_dim: int
    num_hidden_layers: int
    max_n_ply: int = 201
    strategy: bool = False

    def create_model(self) -> 'Transformer':
        return Transformer(self)

    def create_caching_model(self) -> 'TransformerWithCache':
        return TransformerWithCache(self)


class Embeddings(nn.Module):
    embed_dim: int
    piece_type: int = 5
    n_pieces: int = 16
    board_size: int = 7
    max_n_ply: int = 201
    strategy_features: int = None

    @nn.compact
    def __call__(self, tokens: jnp.ndarray, eval: bool):
        embeddings = nn.Embed(self.piece_type, self.embed_dim)(tokens[..., 0])
        embeddings += nn.Embed(self.n_pieces, self.embed_dim)(tokens[..., 1])
        embeddings += nn.Embed(self.board_size, self.embed_dim)(tokens[..., 2])
        embeddings += nn.Embed(self.board_size, self.embed_dim)(tokens[..., 3])
        embeddings += nn.Embed(self.max_n_ply, self.embed_dim)(jnp.clip(tokens[..., 4], 0, self.max_n_ply - 1))

        if self.strategy_features is not None:
            embeddings += nn.Embed(3, self.embed_dim)(tokens[..., 5])
            embeddings += nn.Embed(5, self.embed_dim)(tokens[..., 6])

        embeddings = nn.LayerNorm(epsilon=1e-12)(embeddings)
        embeddings = nn.Dropout(0.5, deterministic=eval)(embeddings)

        return embeddings


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
    def __call__(self, x, vk_cache):
        # x: [EmbedDim]
        # v: [SeqLen k, Head, Dim]
        # k: [SeqLen k, Head, Dim]
        head_dim = self.embed_dim // self.num_heads

        # [1, Head, Dim]
        v_i = nn.Dense(features=self.embed_dim)(x).reshape(1, self.num_heads, head_dim)
        q_i = nn.Dense(features=self.embed_dim)(x).reshape(self.num_heads, head_dim)
        k_i = nn.Dense(features=self.embed_dim)(x).reshape(1, self.num_heads, head_dim)

        # [SeqLen k, Head, Dim]
        v = jnp.concatenate((vk_cache[0, 1:], v_i), axis=0)
        k = jnp.concatenate((vk_cache[1, 1:], k_i), axis=0)

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

        return out_i, jnp.stack([v, k])


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

    def setup(self):
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

    def setup(self):
        self.attention = MultiHeadAttentionWithCache(
            num_heads=self.num_heads,
            embed_dim=self.embed_dim)

        self.feed_forward = FeedForward(embed_dim=self.embed_dim)

    @nn.compact
    def __call__(self, x, cache, eval):
        out, cache = self.attention(x, cache)

        x = x + out
        x = nn.LayerNorm()(x)
        x = x + self.feed_forward(x, eval)
        x = nn.LayerNorm()(x)

        return x, cache


class Transformer(nn.Module):
    config: TransformerConfig

    def __hash__(self):
        return hash(self.config)

    def setup(self):
        self.embeddings = Embeddings(
            self.config.embed_dim,
            max_n_ply=self.config.max_n_ply,
            strategy_features=5 if self.config.strategy else None
        )

        self.layers = [TransformerBlock(self.config.num_heads, self.config.embed_dim)
                       for _ in range(self.config.num_hidden_layers)]

    @nn.compact
    def __call__(self, x: jnp.ndarray, eval=True):
        # [Batch, 1, SeqLen, SeqLen]
        mask = nn.make_causal_mask(jnp.zeros((x.shape[0], x.shape[1])), dtype=bool)

        x = self.embeddings(x, eval)

        for i in range(self.config.num_hidden_layers):
            x = self.layers[i](x, mask, eval=eval)

        x = nn.Dropout(0.1, deterministic=eval)(x)

        p = nn.Dense(features=32, name="head_p")(x)
        v = nn.Dense(features=7, name="head_v")(x)
        c = nn.Dense(features=8, name="head_c")(x)

        return p, v, c  # [Batch, SeqLen, ...]


class TransformerWithCache(nn.Module):
    config: TransformerConfig

    def __hash__(self):
        return hash(self.config)

    def setup(self):
        self.embeddings = Embeddings(
            self.config.embed_dim,
            max_n_ply=self.config.max_n_ply,
            strategy_features=5 if self.config.strategy else None
        )

        self.layers = [
            TransformerBlockWithCache(self.config.num_heads, self.config.embed_dim)
            for _ in range(self.config.num_hidden_layers)
        ]

    def create_cache(self, seq_len):
        head_dim = self.config.embed_dim // self.config.num_heads

        # [2, Layer, SeqLen, Head, HeadDim]
        cache = jnp.zeros((self.config.num_hidden_layers, 2, seq_len, self.config.num_heads, head_dim))
        return cache

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        cache: jnp.ndarray,
        eval=True
    ):
        if x.shape[0] == 5:
            x = self.embeddings(x, eval)

        elif x is None:
            x = jnp.zeros(self.config.embed_dim)

        for i, layer in enumerate(self.layers):
            x, cache_i = layer(x, cache[i], eval=eval)
            cache = cache.at[i].set(cache_i)

        x = nn.Dropout(0.1, deterministic=eval)(x)

        p = nn.Dense(features=32, name="head_p")(x)
        v = nn.Dense(features=7, name="head_v")(x)
        c = nn.Dense(features=8, name="head_c")(x)

        return x, p, v, c, cache


class TrainStateTransformer(TrainStateBase):
    @partial(jax.jit, static_argnames=['eval'])
    def train_step(
        self, x: jnp.ndarray, eval: bool
    ) -> tuple[TrainStateTransformer, jnp.ndarray, jnp.ndarray]:
        tokens, p_true, v_true, c_true = FORMAT_X7ARC.astuple(x)

        if not eval:
            (loss, losses), grads = jax.value_and_grad(loss_fn, has_aux=True)(
                self.params, self, tokens, p_true, v_true, c_true, self.dropout_rng, eval=eval
            )
            state = self.apply_gradients(grads=grads, dropout_rng=random.PRNGKey(self.epoch))
        else:
            loss, losses = loss_fn(
                self.params, self, tokens, p_true, v_true, c_true, self.dropout_rng, eval=eval
            )
            state = self

        return state, loss, losses

    def get_head_names(self) -> list[str]:
        return ['P', 'V', 'C']


@jax.jit
def calc_loss(
    x: jnp.ndarray,
    p_pred: jnp.ndarray, v_pred: jnp.ndarray, c_pred: jnp.ndarray,
    p_true: jnp.ndarray, v_true: jnp.ndarray, c_true: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:

    mask = jnp.any(x != 0, axis=-1)
    mask = mask.reshape(-1)

    # [Batch, SeqLen, 144]
    p_true = p_true.reshape(-1)
    v_true = jnp.stack([v_true]*v_pred.shape[-2], axis=-1).reshape(-1)
    c_true = jnp.stack([c_true]*c_pred.shape[-2], axis=-1).reshape(-1, 8)
    # c_true = c_true.reshape(-1, 1, 8)

    p_pred = p_pred.reshape(-1, 32)
    v_pred = v_pred.reshape(-1, 7)
    c_pred = c_pred.reshape(-1, 8)

    loss_p = optax.softmax_cross_entropy_with_integer_labels(p_pred, p_true)
    loss_v = optax.softmax_cross_entropy_with_integer_labels(v_pred, v_true)
    loss_c = optax.sigmoid_binary_cross_entropy(c_pred, c_true).mean(axis=-1)

    loss_p = jnp.average(loss_p, weights=mask)
    loss_v = jnp.average(loss_v, weights=mask)
    loss_c = jnp.average(loss_c, weights=mask)

    loss = loss_p + loss_v + loss_c
    losses = jnp.array([loss_p, loss_v, loss_c])

    return loss, losses


@partial(jax.jit, static_argnames=['eval'])
def loss_fn(
    params,
    state: TrainStateTransformer,
    tokens: jnp.ndarray,
    p_true: jnp.ndarray,
    v_true: jnp.ndarray,
    c_true: jnp.ndarray,
    dropout_rng,
    eval: bool
) -> tuple[jnp.ndarray, tuple[jnp.ndarray, jnp.ndarray]]:
    # p, v, c = state.apply_fn({'params': params}, tokens, eval=eval, rngs={'dropout': dropout_rng})
    p, v, c = state.apply_fn({'params': params}, tokens, eval=eval, rngs={'dropout': dropout_rng})
    loss, losses = calc_loss(tokens, p, v, c, p_true, v_true, c_true)

    return loss, losses


def create_concat_input(tokens: jnp.ndarray, pos: jnp.ndarray, c_true: jnp.ndarray):
    n_cap_1 = ((pos[..., 8:] == 36) * (c_true[..., None, :] == 0)).sum(axis=-1)
    n_cap_2 = ((pos[..., 8:] == 36) * (c_true[..., None, :] == 1)).sum(axis=-1)

    n_cap_1 = jax.nn.one_hot(n_cap_1, num_classes=4)
    n_cap_2 = jax.nn.one_hot(n_cap_2, num_classes=4)

    t = (tokens[..., 4] % 2 == 0) ^ (tokens[..., 8, :1] < 2)
    t = jax.nn.one_hot(t.astype(jnp.uint8), num_classes=2)

    return jnp.concatenate([n_cap_1, n_cap_2, t], axis=-1)


def test_performance():
    import jax.random
    import time
    from batch import load, get_tokens

    model = Transformer(num_heads=4, embed_dim=256, num_hidden_layers=4, length_memory_block=4)
    model_caching = TransformerWithCache(num_heads=4, embed_dim=256, num_hidden_layers=4, length_memory_block=4)

    variables = model.init(jax.random.PRNGKey(0), jnp.zeros((1, 200, 5), dtype=jnp.uint8))

    batch = load('./data/replay_buffer/run-3.npz')
    tokens = get_tokens(batch)

    cache = model_caching.create_cache(240)
    mem = model_caching.create_zero_memory()

    for i in range(len(mem)):
        _, _, _, _, cache = model_caching.apply(
            variables, mem[i], cache=cache, read_memory_i=jnp.array(i), eval=True
        )

    start = time.perf_counter()

    for seq_i in range(tokens.shape[1]):
        for i in range(tokens.shape[-2]):
            if jnp.all(tokens[0, 0, i] == 0):
                break

            _, _, v, _, cache = model_caching.apply(
                variables, tokens[0, 0, i], cache=cache, eval=True
            )
            print(seq_i, i, v.sum())
    print(time.perf_counter() - start)


if __name__ == '__main__':
    test_performance()
