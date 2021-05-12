import os
import time

import gym
import numpy as np
from gym import spaces


class TicTacToeEnv(gym.Env):
    def __init__(self, opponent="moderate"):
        self.opponent = opponent
        self.episode = 0
        self.observation_space = spaces.Box(low=-1, high=1, shape=(9,), dtype=np.int)
        self.action_space = spaces.Discrete(9)

    def reset(self):
        self.episode += 1
        self.total_reward = 0
        self.occupied = 0
        self.board = np.zeros((3, 3))
        state = self.board.flatten()
        return state

    def step(self, action):

        # Convert action into board position
        row = action // 3
        col = action % 3

        # If agent picks an occupied space repeated end the game and give a penalty
        # Otherwise, give a small penalty and try again
        if self.board[row, col] != 0:
            self.occupied += 1
            if self.occupied > 10:
                reward = -1
                self.total_reward += reward
                self.save_board(self.total_reward)
                return self.board.flatten(), reward, True, {"reward": reward}
            else:
                reward = -0.1
                self.total_reward += reward
                return self.board.flatten(), reward, False, {"reward": reward}
        else:
            self.occupied = 0

        # Otherwise agent actions action updates the board and check for a win
        self.board[row, col] = 1
        if check_win(self.board) == 1:
            reward = 1
            self.total_reward += reward
            self.save_board(self.total_reward)
            return self.board.flatten(), reward, True, {"reward": reward}

        # Check if last move
        if (self.board != 0).all():
            reward = 0
            self.total_reward += reward
            self.save_board(self.total_reward)
            return self.board.flatten(), reward, True, {"reward": reward}
        # If not then opponent moves
        else:
            self.move_opponent()
            if check_win(self.board) == -1:
                reward = -1
                self.total_reward += reward
                self.save_board(self.total_reward)
                return self.board.flatten(), reward, True, {"reward": reward}

        return self.board.flatten(), 0, False, {"reward": 0}

    def move_opponent(self):
        if self.opponent == "random":
            options = np.argwhere(self.board == 0)
            idx = np.random.randint(options.shape[0])
            self.board[tuple(options[idx])] = -1
        elif self.opponent == "moderate":
            move = None
            options = np.argwhere(self.board == 0)
            if np.random.rand() < 0.1:
                idx = np.random.randint(options.shape[0])
                self.board[tuple(options[idx])] = -1
            else:
                # Check if there's a next move that could win
                for o in options:
                    board = self.board.copy()
                    board[tuple(o)] = -1
                    if check_win(board) == -1:
                        move = tuple(o)
                        break
                # Otherwise check for a block
                if not move:
                    for o in options:
                        board = self.board.copy()
                        board[tuple(o)] = 1
                        if check_win(board) == 1:
                            move = tuple(o)
                            break
                # Otherwise, take a random option
                if not move:
                    idx = np.random.randint(options.shape[0])
                    move = tuple(options[idx])
                self.board[move] = -1

    def save_board(self, reward, path="/opt/ml/output/data/"):
        np.save(
            os.path.join(path, "episode_{}_reward_{}.npy".format(self.episode, reward)), self.board
        )


def check_win(board):
    v = board.sum(axis=0)
    h = board.sum(axis=1)
    dd = board[0, 0] + board[1, 1] + board[2, 2]
    du = board[2, 0] + board[1, 1] + board[0, 2]

    if max(v.max(), h.max()) == 3 or dd == 3 or du == 3:
        return 1
    elif min(v.min(), h.min()) == -3 or dd == -3 or du == -3:
        return -1
    else:
        return 0
