from players.config import PlayerConfig, PlayerRandomConfig, PlayerNaotti2020Config, PlayerMCTSConfig
from players.base import PlayerBase
from players.naotti import PlayerNaotti2020
from players.simple import PlayerRandom
from players.mcts import PlayerMCTS


PLAYER_DICT = {
    PlayerRandomConfig: PlayerRandom.from_config,
    PlayerNaotti2020Config: PlayerNaotti2020.from_config,
    PlayerMCTSConfig: PlayerMCTS.from_config,
}


def create_player(config: PlayerConfig, project_dir: str) -> PlayerBase:
    create_func = PLAYER_DICT[type(config)]
    return create_func(config, project_dir)
