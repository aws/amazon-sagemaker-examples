import numpy as np
import pygame


class TicTacToeView:


    def __init__(self):

        self.game_over = False
        self.screen_size = (900, 500)
        pygame.init()
        pygame.font.init()
        pygame.display.set_caption("Tic Tac Toe")
        self.screen = pygame.display.set_mode(self.screen_size)

        # Create a background
        self.background = pygame.Surface(self.screen.get_size()).convert()
        self.background.fill((0, 0, 0))

        # Create a layer for the game
        self.game_surface = pygame.Surface(self.screen.get_size()).convert_alpha()
        self.game_surface.fill((0, 0, 0))


    def reset_game(self):
        self.game_over = False


    def update(self, board, mode='human'):
        try:
            img = self.__draw_game(mode, board)
            self.__handle_pygame_events()
        except Exception as e:
            self.game_over = True
            pygame.display.quit()
            pygame.quit()
            raise e
        else:
            return img


    def __draw_game(self, mode, board):

        self.game_surface.fill((0, 0, 0))

        self.__draw_board(board)

        # update the screen
        self.screen.blit(self.background, (0, 0))
        self.screen.blit(self.game_surface, (0, 0))
        if mode == 'human':
            pygame.display.flip()

        return np.flipud(np.rot90(pygame.surfarray.array3d(pygame.display.get_surface())))


    def __handle_pygame_events(self):
        if not self.game_over:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.game_over = True
                    pygame.display.quit()
                    pygame.quit()


    def __draw_board(self, board):

        pixels = 100

        Xs = np.argwhere(board == 1)
        for X in Xs:
            pygame.draw.rect(self.game_surface, (255, 0, 0), (X[0] * pixels, X[1] * pixels, pixels, pixels))

        Os = np.argwhere(board == -1)
        for O in Os:
            pygame.draw.rect(self.game_surface, (0, 0, 255), (O[0] * pixels, O[1] * pixels, pixels, pixels))
