from jax import numpy as jnp
from flax import linen as nn


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
    def __call__(self, x, cache1, cache2, eval):
        out, cache1, cache2 = self.attention(x, cache1, cache2)

        x = x + out
        x = nn.LayerNorm()(x)
        x = x + self.feed_forward(x, eval)
        x = nn.LayerNorm()(x)

        return x, cache1, cache2


class TransformerDecoder(nn.Module):
    num_heads: int
    embed_dim: int
    num_hidden_layers: int

    max_n_ply: int = 200

    def setup(self):
        self.embeddings = Embeddings(self.embed_dim, max_n_ply=self.max_n_ply)

        self.layers = [TransformerBlock(self.num_heads, self.embed_dim)
                       for _ in range(self.num_hidden_layers)]

    @nn.compact
    def __call__(self, x, eval=True):
        # [Batch, 1, SeqLen, SeqLen]
        mask = nn.make_causal_mask(jnp.zeros((x.shape[0], x.shape[1])), dtype=bool)
        x = self.embeddings(x, eval)

        for i in range(self.num_hidden_layers):
            x = self.layers[i](x, mask, eval=eval)
        x = nn.Dropout(0.1, deterministic=eval)(x)

        pi = nn.Dense(features=32)(x)
        v = nn.Dense(features=7)(x)
        color = nn.Dense(features=8)(x)

        return pi, v, color  # [Batch, SeqLen, ...]


class TransformerDecoderWithCache(nn.Module):
    num_heads: int
    embed_dim: int
    num_hidden_layers: int

    max_n_ply: int = 200

    def setup(self):
        self.embeddings = Embeddings(self.embed_dim, max_n_ply=self.max_n_ply)
        self.layers = [TransformerBlockWithCache(self.num_heads, self.embed_dim)
                       for _ in range(self.num_hidden_layers)]

    def create_cache(self, seq_len):
        head_dim = self.embed_dim // self.num_heads

        # [Batch, Layer, SeqLen, Head, HeadDim]
        v = jnp.zeros((self.num_hidden_layers, seq_len, self.num_heads, head_dim))
        k = jnp.zeros((self.num_hidden_layers, seq_len, self.num_heads, head_dim))

        return v, k

    @nn.compact
    def __call__(self, x, cache1, cache2, eval=True):
        x = self.embeddings(x, eval)

        next_cache1, next_cache2 = cache1, cache2

        i = 0
        for layer in self.layers:
            x, cache1_i, cache2_i = layer(x, cache1[i], cache2[i], eval=eval)

            next_cache1 = next_cache1.at[i].set(cache1_i)
            next_cache2 = next_cache2.at[i].set(cache2_i)

            i += 1

        x = nn.Dropout(0.1, deterministic=eval)(x)

        logits_pi = nn.Dense(features=32)(x)
        logits_v = nn.Dense(features=7)(x)
        logits_color = nn.Dense(features=8)(x)

        return logits_pi, logits_v, logits_color, next_cache1, next_cache2
