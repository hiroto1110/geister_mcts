import socket
import time

import click
import numpy as np

from players.mcts import PlayerMCTS
from players.config import SearchParameters
import env.state as game
import game_analytics
import server_util

from network.checkpoints import Checkpoint

DIRECTION_DICT = {-6: 0, -1: 1, 1: 2, 6: 3}


def connect_and_recv(s: socket.socket, address, num_max_retry=10) -> str:
    for _ in range(num_max_retry):
        try:
            s.connect(address)
            return recv(s)
        except ConnectionRefusedError | ConnectionResetError:
            time.sleep(1)
            continue

    raise ConnectionRefusedError()


def send(s: socket.socket, msg: str):
    print(f'SEND:[{msg.rstrip()}]')
    s.sendall(msg.encode())


def recv(s: socket.socket) -> str:
    msg = s.recv(2**12).decode()
    print(f'RECV:[{msg.rstrip()}]')

    return msg


class Client:
    def __init__(self, name: str, ckpt: Checkpoint, mcts_params: SearchParameters) -> None:
        self.name = name
        self.ckpt = ckpt
        self.mcts_params = mcts_params

        self.player = PlayerMCTS(
            self.ckpt.params,
            self.ckpt.model.create_caching_model(),
            self.mcts_params
        )

        self.memories: list[np.ndarray] = []
        self.win_count = [0, 0, 0]

    def init_state(self, color: np.ndarray, player: int):
        self.state = game.SimulationState(color, player)

        self.player.init_state(self.state)

        if self.player.memory is not None:
            self.memories.append(self.player.memory.reshape(-1))

    def show_memories(self):
        import matplotlib.pyplot as plt
        from sklearn.manifold import TSNE

        memories = TSNE(n_components=2, random_state=0, perplexity=5).fit_transform(np.stack(self.memories))

        plt.scatter(memories[:, 0], memories[:, 1], c=np.linspace(0, 1, len(memories)))
        plt.colorbar()
        plt.show()

    def select_next_action(self):
        if not self.state.is_done:
            return self.player.select_next_action()

        assert self.state.win_type == game.WinType.ESCAPE

        for i in range(2):
            escaping_pos = self.state.escape_pos_p[i]

            d_id = 1 if escaping_pos % 6 == 0 else 2

            escaped = escaping_pos == self.state.pieces_p
            if np.any(escaped):
                p_id = np.where(escaped)[0][0]
                action = p_id * 4 + d_id
                return action

        assert False

    def apply_player_action(self, action, cap_color):
        tokens, afterstates = self.state.step(action, 1)
        self.state.undo_step(action, 1, tokens, afterstates)

        color = np.zeros(8, dtype=np.uint8)

        for i in range(len(afterstates)):
            afterstate = afterstates[i]

            if afterstate.type == game.AfterstateType.CAPTURING:
                color[afterstate.piece_id] = cap_color

        self.player.apply_action(action, player=1, true_color_o=color)

    def apply_opponent_action(self, action):
        self.player.apply_action(action, player=-1, true_color_o=None)

    def calc_opponent_action(self, pieces):
        if np.all(pieces == self.state.pieces_o):
            return -1

        p_id = np.where(pieces != self.state.pieces_o)[0][0]

        d = pieces[p_id] - self.state.pieces_o[p_id]
        d_id = DIRECTION_DICT[d]

        return p_id * 4 + d_id

    def print_board(self):
        color = self.player.node.predicted_color
        if color is None:
            color = np.array([0.5]*8)

        s = game_analytics.state_to_str(self.state, color, colored=True)
        print(s)

    def connect_and_start(self, ip: str, port: int):
        print(f"Connecting to: ({ip}, {port})")

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            _ = connect_and_recv(sock, (ip, port))
            self.start(sock)

    def start(self, sock: socket.socket):
        set_msg = server_util.encode_set_message(self.state.color_p, self.state.root_player, name=self.name)
        send(sock, set_msg)

        recv(sock)
        board_msg = recv(sock)

        prev_t = time.perf_counter()

        while True:
            pieces_o, _ = server_util.parse_board_str(board_msg, self.state.root_player)
            action_o = self.calc_opponent_action(pieces_o)

            if action_o >= 0:
                self.apply_opponent_action(action_o)

            print()
            self.print_board()

            action = self.select_next_action()

            action_msg = server_util.format_action_message(action, self.state.root_player)
            send(sock, action_msg)

            action_responce = recv(sock)

            is_done, winner = server_util.is_done_message(action_responce)
            if is_done:
                self.win_count[winner + 1] += 1
                break

            board_msg = recv(sock)

            is_done, winner = server_util.is_done_message(board_msg)
            if is_done:
                self.win_count[winner + 1] += 1
                break

            color = server_util.parse_action_ack(action_responce)
            self.apply_player_action(action, color)

            self.print_board()

            print()
            print(f"elpased: {time.perf_counter() - prev_t}")
            prev_t = time.perf_counter()


@click.command()
@click.argument('ip', type=str)
@click.argument('first_player', type=bool)
@click.argument('num_games', type=int)
@click.argument('player_name', type=str)
@click.argument('ckpt_path', type=str)
def main(
    ip: str,
    first_player: bool,
    num_games: int,
    player_name: str,
    ckpt_path: str
):
    ckpt = Checkpoint.from_json_file(ckpt_path)

    search_params = SearchParameters(
        num_simulations=400,
        dirichlet_alpha=0.1,
        n_ply_to_apply_noise=2,
        depth_search_checkmate_leaf=6,
        depth_search_checkmate_root=9,
        max_duplicates=2,
        visibilize_node_graph=False
    )

    client = Client(player_name, ckpt, search_params)

    for k in range(2):
        if first_player:
            port = 10000 + (k + 0) % 2
        else:
            port = 10000 + (k + 1) % 2

        player = 1 if port == 10000 else -1

        for i in range(num_games):
            client.init_state(game.create_random_color(), player)
            client.connect_and_start(ip, port)

            print("win: {2}, draw: {1}, lost: {0}".format(*client.win_count))

    # client.show_memories()


if __name__ == '__main__':
    main()
