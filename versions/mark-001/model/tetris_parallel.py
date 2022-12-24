from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tf_agents.environments import parallel_py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts


from model.board import Board

class Tetris(parallel_py_environment.ParallelPyEnvironment):

  def __init__(self, WINDOW_SIZE=(288, 528), PADDING=(24, 24), BOARD=(10, 20), BLOCK_SIZE=(24, 24)):
    self.WINDOW_SIZE = WINDOW_SIZE
    self.PADDING = PADDING
    self.BOARD = BOARD
    self.BLOCK_SIZE = BLOCK_SIZE
    
    self.init_board()

    # 0: left, 1: right, 2: rotate (clockwise), 3: drop
    self._action_spec = array_spec.BoundedArraySpec(shape=(), dtype=np.int32, minimum=0, maximum=3, name='action')
    # self._observation_spec = array_spec.BoundedArraySpec(shape=(200, ), dtype=np.int32, minimum=0, maximum=1, name='board')
    self._observation_spec = array_spec.BoundedArraySpec(shape=(24, ), dtype=np.int32, minimum=-2, maximum=20, name='board')

  def init_board(self):
    self.board = Board(self.WINDOW_SIZE, self.PADDING, self.BOARD, self.BLOCK_SIZE)
    self._state = self.board.state
    self._episode_ended = False

  def render(self):
    return self.board.render()

  def _reset(self):
    self.init_board()
    return ts.restart(self._state)

  def action_spec(self):
    return self._action_spec

  def observation_spec(self):
    return self._observation_spec

  def _step(self, action):
    if self._episode_ended:
      return self.reset()

    # Make sure episodes don't go on forever.
    # 0: left, 1: right, 2: rotate (clockwise), 3: drop
    if action in [0, 1, 2, 3]:
        is_ended, reward, state = self.board.take_action(action)
        self._episode_ended = is_ended
        self._state = state
    
    else:
        raise ValueError(f'Unrecognized action for "{action}"')

    if self._episode_ended:
      return ts.termination(self._state, reward)

    else:
        return ts.transition(self._state, reward, discount=1)
