import socket
import numpy as np
import orbax.checkpoint

import mcts
import geister as game
import game_analytics
import server_util
from network_transformer import TransformerDecoderWithCache

DIRECTION_DICT = {-6: 0, -1: 1, 1: 2, 6: 3}


def send(s: socket.socket, msg: str):
    print(f'SEND:[{msg.rstrip()}]')
    s.send(msg.encode())


def recv(s: socket.socket) -> str:
    msg = s.recv(2**12).decode()
    print(f'RECV:[{msg.rstrip()}]')

    return msg


class Client:
    def __init__(self, model, params, search_params: mcts.SearchParameters) -> None:
        self.model = model
        self.pred_state = mcts.PredictState(self.model.apply, params)

        self.search_params = search_params
        self.win_count = [0, 0, 0]

    def init_state(self, color: np.ndarray, player: int):
        self.state = game.SimulationState(color, player)
        self.node, _ = mcts.create_root_node(self.state, self.pred_state, self.model)

    def select_next_action(self):
        if not self.state.is_done:
            return mcts.select_action_with_mcts(self.node, self.state, self.pred_state, self.search_params)

        assert self.state.win_type == game.WinType.ESCAPE

        for i in range(2):
            if self.state.root_player == 1:
                escaping_pos = self.state.escape_pos_p[i]
            else:
                escaping_pos = self.state.escape_pos_o[i]

            d_id = 1 if escaping_pos % 6 == 0 else 2

            escaped = escaping_pos == self.state.pieces_p
            if escaped.any():
                p_id = np.where(escaped)[0][0]
                action = p_id * 4 + d_id
                return action

        assert False

    def apply_player_action(self, action, color):
        tokens, afterstates = self.state.step(action, 1)

        if self.state.is_done:
            return

        if len(afterstates) > 0:
            for i in range(len(afterstates)):
                afterstate = afterstates[i]
                color_i = color if afterstate.type == game.AfterstateType.CAPTURING else 0

                child, _ = mcts.expand_afterstate(self.node, tokens, afterstates[i:],
                                                  self.state, self.pred_state, self.search_params)
                tokens_afterstate = self.state.step_afterstate(afterstate, color_i)
                tokens += tokens_afterstate

            self.node, _ = mcts.expand(child, tokens_afterstate, self.state, self.pred_state, self.search_params)
        else:
            self.node, _ = mcts.expand(self.node, tokens, self.state, self.pred_state, self.search_params)

    def apply_opponent_action(self, action):
        tokens, _ = self.state.step(action, -1)
        self.node, _ = mcts.expand(self.node, tokens, self.state, self.pred_state, self.search_params)

    def calc_opponent_action(self, pieces):
        if np.all(pieces == self.state.pieces_o):
            return -1

        print(pieces, self.state.pieces_o)
        p_id = np.where(pieces != self.state.pieces_o)[0][0]

        d = pieces[p_id] - self.state.pieces_o[p_id]
        d_id = DIRECTION_DICT[d]

        return p_id * 4 + d_id

    def print_board(self):
        color = self.node.predicted_color
        if color is None:
            color = np.array([0.5]*8)

        s = game_analytics.state_to_str(self.state, color, colored=True)
        print(s)

    def connect_and_start(self, ip: str, port: int):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.connect((ip, port))
            self.start(sock)

    def start(self, sock: socket.socket):
        set_msg = recv(sock)

        set_msg = server_util.encode_set_message(self.state.color_p, self.state.root_player)
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


def main(ip='127.0.0.1',
         port=10000):

    ckpt_dir = './checkpoints/run-2'

    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    checkpoint_manager = orbax.checkpoint.CheckpointManager(ckpt_dir, orbax_checkpointer)

    ckpt = checkpoint_manager.restore(6500)

    params = ckpt['state']['params']
    model = TransformerDecoderWithCache(**ckpt['model'])

    search_params = mcts.SearchParameters(num_simulations=50,
                                          dirichlet_alpha=0.1,
                                          n_ply_to_apply_noise=0,
                                          depth_search_checkmate_leaf=4,
                                          depth_search_checkmate_root=8,
                                          max_duplicates=1,
                                          should_do_visibilize_node_graph=False)

    client = Client(model, params, search_params)

    if port == 10000:
        player = 1
    else:
        player = -1

    for i in range(10):
        try:
            # client.init_state(np.array([0, 0, 0, 0, 1, 1, 1, 1]))
            client.init_state(game.create_random_color(), player)
            client.connect_and_start(ip, port)
        except Exception as e:
            raise e
            print(e)


if __name__ == '__main__':
    main()
