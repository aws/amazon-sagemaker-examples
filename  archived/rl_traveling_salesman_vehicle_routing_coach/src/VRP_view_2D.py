import numpy as np
import pygame


class VRPView2D:
    def __init__(self, n_restaurants, n_orders, map_quad, grid_size):

        self.n_orders = n_orders
        self.n_restaurants = n_restaurants
        self.grid_size = grid_size

        # map boundaries
        self.map_min_x = -map_quad[0]
        self.map_max_x = +map_quad[0]
        self.map_min_y = -map_quad[1]
        self.map_max_y = +map_quad[1]

        pygame.init()
        pygame.display.set_caption("VRP")

        self.screen = pygame.display.set_mode(self.__get_window_size())

        # Create a background
        self.background = pygame.Surface(self.screen.get_size()).convert()
        self.background.fill((0, 0, 0))

        # Create a layer for the game
        self.game_surface = pygame.Surface(self.screen.get_size()).convert_alpha()
        self.game_surface.fill(
            (
                0,
                0,
                0,
                0,
            )
        )

    def update(self, res_x, res_y, o_status, o_x, o_y, dr_x, dr_y, mode="human"):
        try:
            self.game_surface.fill(
                (
                    0,
                    0,
                    0,
                    0,
                )
            )

            self.__draw_restaurant(res_x, res_y)
            self.__draw_orders(o_status, o_x, o_y)
            self.__draw_driver(dr_x, dr_y)

            # update the screen
            self.screen.blit(self.background, (0, 0))
            self.screen.blit(self.game_surface, (0, 0))
            self.__draw_legend()

            if mode == "human":
                pygame.display.flip()

            img = np.flipud(np.rot90(pygame.surfarray.array3d(pygame.display.get_surface())))

            self.__handle_pygame_events()
        except Exception as e:
            pygame.display.quit()
            pygame.quit()
            raise e
        else:
            return img

    def __handle_pygame_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.game_over = True
                pygame.display.quit()
                pygame.quit()

    def __get_window_size(self):
        # Add space for legend
        win_w = (self.map_max_x - self.map_min_x + 3) * self.grid_size + (5 * self.grid_size)
        win_h = (self.map_max_y - self.map_min_y + 3) * self.grid_size
        return win_w, win_h

    def __locate_on_window(self, x, y):
        w_x = (x - self.map_min_x + 1) * self.grid_size
        w_y = (self.map_max_y - y + 1) * self.grid_size
        return w_x, w_y

    def __draw_restaurant(self, res_x, res_y):
        for r in range(self.n_restaurants):
            rx = res_x[r]
            ry = res_y[r]
            rgb = (0, 0, 255)
            rxw, ryw = self.__locate_on_window(rx, ry)
            pygame.draw.rect(self.game_surface, rgb, (rxw - 10, ryw - 10, 20, 20))

    def __draw_orders(self, o_status, o_x, o_y):
        # Draw the orders
        for o in range(self.n_orders):
            if o_status[o] != 0:
                ox = o_x[o]
                oy = o_y[o]
                oxw, oyw = self.__locate_on_window(ox, oy)
                if o_status[o] == 1:
                    rgb = (255, 0, 0)
                elif o_status[o] == 2:
                    rgb = (255, 165, 0)
                elif o_status[o] == 3:
                    rgb = (0, 255, 0)
                else:
                    rgb = (255, 255, 255)
                pygame.draw.rect(self.game_surface, rgb, (oxw - 4, oyw - 4, 8, 8))

    def __draw_driver(self, dr_x, dr_y):
        dxw, dyw = self.__locate_on_window(dr_x, dr_y)
        pygame.draw.rect(self.game_surface, (255, 255, 0), (dxw - 7, dyw - 7, 14, 14))

    def __draw_legend(self):
        # Driver
        ax, ay = self.__locate_on_window(self.map_max_x + 2, self.map_max_y)
        pygame.draw.rect(self.game_surface, (255, 255, 0), (ax - 7, ay - 7, 14, 14))
        self.screen.blit(self._create_text_surface("Driver"), (ax + 10, ay - 7))

        # Restaurant
        rx, ry = self.__locate_on_window(self.map_max_x + 2, self.map_max_y - 2)
        pygame.draw.rect(self.game_surface, (0, 0, 255), (rx - 10, ry - 10, 20, 20))
        self.screen.blit(self._create_text_surface("Restaurant"), (rx + 12, ry - 10))

        # Orders
        ox, oy = self.__locate_on_window(self.map_max_x + 2, self.map_max_y - 3)
        pygame.draw.rect(self.game_surface, (255, 0, 0), (ox - 4, oy - 4, 8, 8))
        self.screen.blit(self._create_text_surface("Order Open"), (ox + 12, oy - 10))

        ox, oy = self.__locate_on_window(self.map_max_x + 2, self.map_max_y - 4)
        pygame.draw.rect(self.game_surface, (255, 165, 0), (ox - 4, oy - 4, 8, 8))
        self.screen.blit(self._create_text_surface("Order Accepted"), (ox + 12, oy - 10))

        ox, oy = self.__locate_on_window(self.map_max_x + 2, self.map_max_y - 5)
        pygame.draw.rect(self.game_surface, (0, 255, 0), (ox - 4, oy - 4, 8, 8))
        self.screen.blit(self._create_text_surface("Order Picked Up"), (ox + 12, oy - 10))

        self.screen.blit(self.game_surface, (0, 0))

    @staticmethod
    def _create_text_surface(text, color=(238, 130, 238)):
        font = pygame.font.SysFont("arial", 16)
        return font.render(text, True, color)


if __name__ == "__main__":
    vrp = VRPView2D(n_restaurants=2, n_orders=5, map_quad=(10, 10), grid_size=25)
    vrp.update([0, 2], [0, 1], [1, 1, 0, 2, 2], [-2, -2, -1, 1, 2], [-2, -2, -1, 1, 2], 2, 0)
    input("Press any key to quit.")
