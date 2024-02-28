import socket
import numpy as np

from players.mcts import PlayerMCTS
import players.mcts as mcts
import env.state as game
import game_analytics
import server_util

from network.checkpoints import Checkpoint

DIRECTION_DICT = {-6: 0, -1: 1, 1: 2, 6: 3}


def send(s: socket.socket, msg: str):
    print(f'SEND:[{msg.rstrip()}]')
    s.send(msg.encode())


def recv(s: socket.socket) -> str:
    msg = s.recv(2**12).decode()
    print(f'RECV:[{msg.rstrip()}]')

    return msg


class Client:
    def __init__(self, name: str, ckpt: Checkpoint, mcts_params: mcts.SearchParameters) -> None:
        self.name = name
        self.ckpt = ckpt
        self.mcts_params = mcts_params

        self.player = PlayerMCTS(
            self.ckpt.params,
            self.ckpt.model.create_caching_model(),
            self.mcts_params
        )

        self.win_count = [0, 0, 0]

    def init_state(self, color: np.ndarray, player: int):
        self.state = game.SimulationState(color, player)
        self.player.init_state(self.state)

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
            sock.connect((ip, port))
            self.start(sock)

    def start(self, sock: socket.socket):
        while True:
            try:
                set_msg = recv(sock)
                break
            except ConnectionResetError:
                continue

        set_msg = server_util.encode_set_message(self.state.color_p, self.state.root_player, name=self.name)
        send(sock, set_msg)

        recv(sock)
        board_msg = recv(sock)

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


def main(ip='127.0.0.1'):
    ckpt = Checkpoint.from_json_file('./data/projects/run-7/main/300.json')

    search_params = mcts.SearchParameters(
        num_simulations=400,
        dirichlet_alpha=0.1,
        n_ply_to_apply_noise=4,
        depth_search_checkmate_leaf=6,
        depth_search_checkmate_root=8,
        max_duplicates=8,
        visibilize_node_graph=False
    )

    client = Client("Sawagani", ckpt, search_params)

    for k in range(2):
        for i in range(25):
            port = 10000 + (k + 1) % 2

            player = 1 if port == 10000 else -1

            try:
                client.init_state(game.create_random_color(), player)
                client.connect_and_start(ip, port)
            except Exception as e:
                raise e
            finally:
                print("win: {2}, draw: {1}, lost: {0}".format(*client.win_count))


if __name__ == '__main__':
    main()
