import dataclasses

import env.state as game
import env.lib.checkmate_lib as checkmate_lib
import env.lib.checkmate_u_lib as checkmate_u_libc


@dataclasses.dataclass
class CheckmateResult:
    action: int
    eval: int
    escaped_id: int


def find_checkmate(state: game.State, player: int, depth: int) -> CheckmateResult:
    a, e, i = checkmate_lib.find_checkmate(
        state.board[game.POS_P], state.board[game.COL_P],
        state.board[game.POS_O], state.board[game.COL_O],
        player, 1, depth
    )
    return CheckmateResult(a, e, i)