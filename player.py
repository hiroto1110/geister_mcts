import numpy as np
import geister as game


class Player:
    def decide_next_move(self) -> int:
        pass


def play_game(player1: Player, player2: Player, game_length=200, print_board=False):
    player = 1

    state1, state2 = game.get_initial_state_pair()

    for i in range(game_length):
        if player == 1:
            action = player1.decide_next_move()
        else:
            action = player2.decide_next_move()

        state1.step(action, player)
        state2.step(action, -player)

        if print_board:
            board = np.zeros(36, dtype=np.int8)

            board[state1.pieces_p[(state1.pieces_p >= 0) & (state1.color_p == 1)]] = 1
            board[state1.pieces_p[(state1.pieces_p >= 0) & (state1.color_p == 0)]] = 2
            board[state2.pieces_p[(state2.pieces_p >= 0) & (state2.color_p == 1)]] = -1
            board[state2.pieces_p[(state2.pieces_p >= 0) & (state2.color_p == 0)]] = -2

            print(str(board.reshape((6, 6))).replace('0', ' '))
            print(i)

        if state1.is_done or state2.is_done:
            break

        player = -player
