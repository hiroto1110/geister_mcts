import socket
import numpy as np
from flax.training import checkpoints
import orbax.checkpoint

import mcts
import geister as game
from network_transformer import TransformerDecoderWithCache

DIRECTION_DICT = {-6: 0, -1: 1, 1: 2, 6: 3}

COLOR_DICT = {'r': game.RED, 'b': game.BLUE, 'u': game.UNCERTAIN_PIECE}
PIECE_NAMES = 'ABCDEFGH'
DIRECTION_NAMES = ['N', 'W', 'E', 'S']


class Client:
    def __init__(self, model, params, num_sim: int, alpha: float) -> None:
        self.model = model
        self.pred_state = mcts.PredictState(self.model.apply, params)

        self.num_sim = num_sim
        self.alpha = alpha

        self.win_count = [0, 0, 0]

    def send(self, s: str):
        print(f'SEND:[{s.rstrip()}]')
        self.socket.send(s.encode())

    def recv(self) -> str:
        s = self.socket.recv(2**12).decode()
        print(f'RECV:[{s.rstrip()}]')

        return s

    def init_state(self, color: np.ndarray):
        self.state = game.SimulationState(color, -1)
        self.node, _ = mcts.create_root_node(self.state, self.pred_state, self.model)

    def select_next_action(self):
        if not self.state.is_done:
            return mcts.select_action_with_mcts(self.node, self.state, self.pred_state,
                                                self.num_sim, self.alpha,
                                                checkmate_search_depth=10)

        assert self.state.win_type == game.WinType.ESCAPE

        for i in range(2):
            escaping_pos = self.state.escape_pos_p[i]
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

                child, _ = mcts.expand_afterstate(self.node, tokens, afterstates[i:], self.state, self.pred_state)
                tokens_afterstate = self.state.step_afterstate(afterstate, color_i)
                tokens += tokens_afterstate

            self.node, _ = mcts.expand(child, tokens_afterstate, self.state, self.pred_state)
        else:
            self.node, _ = mcts.expand(self.node, tokens, self.state, self.pred_state)

    def apply_opponent_action(self, action):
        tokens, _ = self.state.step(action, -1)
        self.node, _ = mcts.expand(self.node, tokens, self.state, self.pred_state)

    def calc_opponent_action(self, pieces):
        p_id = np.where(pieces != self.state.pieces_o)[0][0]

        d = pieces[p_id] - self.state.pieces_o[p_id]
        d_id = DIRECTION_DICT[d]

        return p_id * 4 + d_id

    def print_board(self):
        board = np.zeros(36, dtype=np.int8)
        board[self.state.pieces_p[(self.state.pieces_p >= 0) & (self.state.color_p == 1)]] = 1
        board[self.state.pieces_p[(self.state.pieces_p >= 0) & (self.state.color_p == 0)]] = 2
        board[self.state.pieces_o[(self.state.pieces_o >= 0) & (self.state.color_o == 1)]] = -1
        board[self.state.pieces_o[(self.state.pieces_o >= 0) & (self.state.color_o == 0)]] = -2
        board[self.state.pieces_o[(self.state.pieces_o >= 0) & (self.state.color_o == 2)]] = -3
        print(str(board.reshape((6, 6))).replace('0', ' '))

    def start(self, ip, port):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((ip, port))

        set_msg = self.recv()

        set_msg = format_set_message(self.state.color_p)
        self.send(set_msg)

        self.recv()
        board_msg = self.recv()

        while True:
            pieces_o, _ = parse_board_str(board_msg)
            action_o = self.calc_opponent_action(pieces_o)
            self.apply_opponent_action(action_o)

            print()
            self.print_board()

            action = self.select_next_action()

            action_msg = format_action_message(action)
            self.send(action_msg)

            action_responce = self.recv()

            is_done, winner = is_done_message(action_responce)
            if is_done:
                self.win_count[winner + 1] += 1
                break

            board_msg = self.recv()

            is_done, winner = is_done_message(board_msg)
            if is_done:
                self.win_count[winner + 1] += 1
                break

            color = parse_action_ack(action_responce)
            self.apply_player_action(action, color)

            self.print_board()

        self.socket.close()


def format_set_message(color: np.ndarray) -> str:
    msg = ''

    for i in range(8):
        if color[i] == game.RED:
            msg += PIECE_NAMES[i]

    return f'SET:{msg}\r\n'


def format_action_message(action: int) -> str:
    p_id = action // 4
    d_id = action % 4

    p_name = PIECE_NAMES[p_id]
    d_id = DIRECTION_NAMES[d_id]

    return f'MOV:{p_name},{d_id}\r\n'


def parse_action_ack(s: str):
    if s[2] == 'R':
        return game.RED

    if s[2] == 'B':
        return game.BLUE

    return game.UNCERTAIN_PIECE


def is_done_message(s: str):
    if s.startswith('WON'):
        return True, 1

    if s.startswith('LST'):
        return True, -1

    if s.startswith('DRW'):
        return True, 0

    return False, 0


def parse_board_str(s: str):
    pieces_o = np.zeros(8, dtype=np.int16)
    color_o = np.zeros(8, dtype=np.int16)

    offset = 24 + 4

    for i in range(8):
        x = int(s[offset + 0 + i*3])
        y = int(s[offset + 1 + i*3])
        c = s[offset + 2 + i*3]

        if x == 9 and y == 9:
            pieces_o[i] = -1
        else:
            pieces_o[i] = y * 6 + x

        color_o[i] = COLOR_DICT[c]

    return pieces_o[::-1], color_o[::-1]


def main(ip='127.0.0.1',
         port=10001,
         num_sim=400,
         alpha=0.2):

    if False:
        ckpt = checkpoints.restore_checkpoint(ckpt_dir='./checkpoints_backup_203/',
                                              prefix='geister_5_256_', target=None)
        params = ckpt['params']

        model = TransformerDecoderWithCache(num_heads=8,
                                            embed_dim=256,
                                            num_hidden_layers=5,
                                            is_linear_attention=False)
    else:
        orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        checkpoint_manager = orbax.checkpoint.CheckpointManager('./checkpoints/driven-bird-204', orbax_checkpointer)

        ckpt = checkpoint_manager.restore(checkpoint_manager.latest_step())

        params = ckpt['state']['params']
        model = TransformerDecoderWithCache(**ckpt['model'])

    client = Client(model, params, num_sim, alpha)

    for i in range(100):
        try:
            client.init_state(np.array([0, 0, 0, 0, 1, 1, 1, 1]))
            client.start(ip, port)
        except Exception as e:
            raise e
        finally:
            print(client.win_count)


if __name__ == '__main__':
    main()
