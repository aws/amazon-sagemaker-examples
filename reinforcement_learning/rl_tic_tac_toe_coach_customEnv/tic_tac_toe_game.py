from functools import partial

import numpy as np
from IPython.display import display
from ipywidgets import HBox, Layout, VBox, widgets


class TicTacToeGame(object):
    """
    Tic-tac-toe game within a Jupyter Notebook
    Opponent is Xs and starts the game.
        This is assumed to be a predictor object from a SageMaker RL trained agent
    """

    def __init__(self, agent):
        self.board = np.zeros((3, 3))
        self.game_over = False
        self.turn = "X"
        self.agent = agent

    def start(self):
        self.board = np.zeros((3, 3))
        self.game_over = False
        self.turn = "X"
        self.draw_board()
        self.move_agent()

    def mark_board(self):
        Xs = np.argwhere(self.board == 1)
        for X in Xs:
            self.spaces[X[0] * 3 + X[1]].description = "X"
        Os = np.argwhere(self.board == -1)
        for O in Os:
            self.spaces[O[0] * 3 + O[1]].description = "O"

    def click_space(self, action, space):

        row = action // 3
        col = action % 3

        if self.game_over:
            return

        if self.board[row, col] != 0:
            self.text_box.value = "Invalid"
            return

        if self.turn == "O":
            self.board[row, col] = -1
            self.mark_board()
            if check_win(self.board) == -1:
                self.text_box.value = "Os Win"
                self.game_over = True
            else:
                self.turn = "X"
                self.text_box.value = "Xs Turn"

        self.move_agent()

    def draw_board(self):

        self.text_box = widgets.Text(value="Xs Turn", layout=Layout(width="100px", height="50px"))
        self.spaces = []

        for i in range(9):
            space = widgets.Button(
                description="",
                disabled=False,
                button_style="",
                tooltip="Click to make move",
                icon="",
                layout=Layout(width="75px", height="75px"),
            )
            self.spaces.append(space)
            space.on_click(partial(self.click_space, i))
        board = VBox(
            [
                HBox([self.spaces[0], self.spaces[1], self.spaces[2]]),
                HBox([self.spaces[3], self.spaces[4], self.spaces[5]]),
                HBox([self.spaces[6], self.spaces[7], self.spaces[8]]),
            ]
        )
        display(VBox([board, self.text_box]))

        return

    def move_agent(self):

        if self.game_over:
            return

        if self.turn == "X":
            # Take the first empty space with the highest preference from the agent
            for action in np.argsort(-np.array(self.agent.predict(self.board.flatten())[1][0])):
                row = action // 3
                col = action % 3
                if self.board[row, col] == 0:
                    self.board[action // 3, action % 3] = 1
                    break
            self.mark_board()
            if check_win(self.board) == 1:
                self.text_box.value = "Xs Win"
                self.game_over = True
            elif (self.board != 0).all():
                self.text_box.value = "Draw"
            else:
                self.turn = "O"
                self.text_box.value = "Os Turn"


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
