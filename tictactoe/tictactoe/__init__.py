from gym.envs.registration import register

register(
    id='tictactoe-v0',
    entry_point='tictactoe.envs:TicTacToeEnv',
)
