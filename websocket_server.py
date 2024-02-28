import asyncio
import websockets
import json
from datetime import datetime

import numpy as np
import orbax.checkpoint
import geister_state as game
import players.mcts as mcts
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
        if player.state.n_ply < 8:
            actions = game.get_valid_actions(player.state, 1)
            return np.random.choice(actions)

        return player.select_next_action()

    if player.state.win_type != game.WinType.ESCAPE:
        return -1

    for i in range(2):
        escaping_pos = player.state.escape_pos_p[i]

        d_id = 0

        escaped = escaping_pos == player.state.pieces_p
        if escaped.any():
            p_id = np.where(escaped)[0][0]
            action = p_id * 4 + d_id
            return action

    return -1


async def echo(websocket, path):
    mcts_params = mcts.SearchParameters(
        num_simulations=50,
        dirichlet_alpha=0.1,
        n_ply_to_apply_noise=0,
        max_duplicates=1,
        depth_search_checkmate_leaf=4,
        depth_search_checkmate_root=8,
        visibilize_node_graph=False
    )

    action_history = []

    try:
        async for message in websocket:
            print(f"RECV: [{message}]")

            if message.startswith("SET"):
                color2, color1 = parse_set_message(message)
                player = mcts.PlayerMCTS(params, model, mcts_params)
                player.init_state(game.SimulationState(color1, -1))

            if message.startswith("MOV"):
                action_o = parse_action_message(message)
                action_history.append(action_o)

                player.apply_action(action_o, 1, color2)

                predicted_color = player.node.predicted_color
                predicted_v = player.node.predicted_v

                print(game_analytics.state_to_str_objectively(
                    player.state.pieces_p, color1,
                    player.state.pieces_o, color2,
                    colored=True))

                action_p = select_next_action(player)
                action_history.append(action_p)

                if action_p == -1:
                    break

                if not player.state.is_done:
                    player.apply_action(action_p, -1, color2)

                print(game_analytics.state_to_str_objectively(
                    player.state.pieces_p, color1,
                    player.state.pieces_o, color2,
                    colored=True))

                def convert_to_list(a):
                    if a is None:
                        return None
                    return [float(f) for f in a]

                msg_data = {
                    "action": int(action_p),
                    "color": convert_to_list(predicted_color),
                    "v": convert_to_list(predicted_v)
                }
                await websocket.send(json.dumps(msg_data))

                if player.state.is_done:
                    break
    except Exception as e:
        raise e

    finally:
        if len(action_history) > 10:
            color = np.concatenate([color1, color2])
            np.savez(f"log/{datetime.now().isoformat()}.npy", actions=np.array(action_history), color=color)


async def main():
    async with websockets.serve(echo, "localhost", 8765):
        await asyncio.Future()


print("start server")
asyncio.run(main())
