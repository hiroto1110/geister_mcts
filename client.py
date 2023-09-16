import socket
import numpy as np

import geister as game


color_dict = {'r': game.RED, 'b': game.BLUE, 'u': game.UNCERTAIN_PIECE}


def parse_board_str(s: str):
    pieces_o = np.zeros(8)
    color_o = np.zeros(8)

    for i in range(8):
        x = int(s[24 + i*3])
        y = int(s[25 + i*3])
        c = s[26 + i*3]

        if x == 9 and y == 9:
            pieces_o[i] = -1
        else:
            pieces_o[i] = y * 6 + x

        color_o[i] = color_dict[c]

    return pieces_o, color_o


def main(ip='127.0.0.1',
         port=10001):
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect((ip, port))

    response = client.recv(2**16)
    print(response)

    client.send('Hello!')


if __name__ == '__main__':
    print(parse_board_str("14R24R34R44R15B25B35B45B41u31u21u11u40u30u20u10u"))
    # main()
