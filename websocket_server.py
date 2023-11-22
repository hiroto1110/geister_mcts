import asyncio
import websockets

import numpy as np
import orbax.checkpoint
import geister as game
import mcts
from network_transformer import TransformerDecoderWithCache
import game_analytics


def parse_set_message(msg):
    color_str = msg[4:]

    color1 = np.zeros(8, dtype=np.int8)
    color2 = np.zeros(8, dtype=np.int8)

    for i in range(8):
        color1[i] = int(color_str[i + 0])
        color2[i] = int(color_str[i + 8])

    return color1, color2


def parse_action_message(msg):
    action_str = msg[4:]
    return int(action_str)


ckpt_dir = './checkpoints/run-2'

orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
checkpoint_manager = orbax.checkpoint.CheckpointManager(ckpt_dir, orbax_checkpointer)

ckpt = checkpoint_manager.restore(6500)

params = ckpt['state']['params']
model = TransformerDecoderWithCache(**ckpt['model'])


def select_next_action(player: mcts.PlayerMCTS):
    if player.state.winner == 0:
        return player.select_next_action()

    assert player.state.win_type == game.WinType.ESCAPE

    for i in range(2):
        escaping_pos = player.state.escape_pos_p[i]

        d_id = 0

        escaped = escaping_pos == player.state.pieces_p
        if escaped.any():
            p_id = np.where(escaped)[0][0]
            action = p_id * 4 + d_id
            return action

    assert False


async def echo(websocket, path):
    mcts_params = mcts.SearchParameters(
        num_simulations=50,
        dirichlet_alpha=0.1,
        n_ply_to_apply_noise=0,
        max_duplicates=1,
        depth_search_checkmate_leaf=4,
        depth_search_checkmate_root=8,
        should_do_visibilize_node_graph=False
    )

    async for message in websocket:
        print(f"RECV: [{message}]")

        if message.startswith("SET"):
            color2, color1 = parse_set_message(message)
            player = mcts.PlayerMCTS(params, model, mcts_params)
            player.init_state(game.SimulationState(color1, -1))

        if message.startswith("MOV"):
            action_o = parse_action_message(message)
            player.apply_action(action_o, 1, color2)

            print(game_analytics.state_to_str_objectively(
                player.state.pieces_p, color1,
                player.state.pieces_o, color2,
                colored=True))

            action_p = select_next_action(player)

            if not player.state.is_done:
                player.apply_action(action_p, -1, color2)

            print(game_analytics.state_to_str_objectively(
                player.state.pieces_p, color1,
                player.state.pieces_o, color2,
                colored=True))

            await websocket.send(f"{action_p}")


async def main():
    async with websockets.serve(echo, "localhost", 8765):
        await asyncio.Future()


print("start server")
asyncio.run(main())
