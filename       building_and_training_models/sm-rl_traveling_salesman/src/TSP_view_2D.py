import numpy as np
import pygame


class TSPView2D:
    def __init__(self, n_orders, map_quad, grid_size):

        self.grid_size = grid_size
        self.state = []

        self.n_orders = n_orders

        self.map_min_x = -map_quad[0]
        self.map_max_x = +map_quad[0]
        self.map_min_y = -map_quad[1]
        self.map_max_y = +map_quad[1]

        pygame.init()
        pygame.display.set_caption("TSP")

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

    def update(
        self,
        agt_at_restaurant,
        restaurant_x,
        restaurant_y,
        o_delivery,
        o_x,
        o_y,
        agt_x,
        agt_y,
        mode="human",
    ):
        try:
            self.game_surface.fill(
                (
                    0,
                    0,
                    0,
                    0,
                )
            )

            self.__draw_restaurant(agt_at_restaurant, restaurant_x, restaurant_y)
            self.__draw_orders(o_delivery, o_x, o_y)
            self.__draw_agent(agt_x, agt_y)

            # update the screen
            self.screen.blit(self.background, (0, 0))
            self.screen.blit(self.game_surface, (0, 0))

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

    def __get_window_size(self):
        win_w = (self.map_max_x - self.map_min_x + 3) * self.grid_size
        win_h = (self.map_max_y - self.map_min_y + 3) * self.grid_size
        return win_w, win_h

    def __locate_on_window(self, x, y):
        w_x = (x - self.map_min_x + 1) * self.grid_size
        w_y = (self.map_max_y - y + 1) * self.grid_size
        return w_x, w_y

    @staticmethod
    def __handle_pygame_events():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.display.quit()
                pygame.quit()

    def __draw_restaurant(self, agt_at_restaurant, restaurant_x, restaurant_y):
        if agt_at_restaurant:
            rgb = (255, 0, 255)
        else:
            rgb = (0, 0, 255)
        rx, ry = self.__locate_on_window(restaurant_x, restaurant_y)
        pygame.draw.rect(self.game_surface, rgb, (rx - 10, ry - 10, 20, 20))

    def __draw_agent(self, agt_x, agt_y):
        axw, ayw = self.__locate_on_window(agt_x, agt_y)
        pygame.draw.rect(self.game_surface, (255, 255, 0), (axw - 7, ayw - 7, 14, 14))
        pass

    def __draw_orders(self, o_delivery, o_x, o_y):
        # Draw the orders
        for i in range(self.n_orders):
            oxw, oyw = self.__locate_on_window(o_x[i], o_y[i])
            if o_delivery[i]:
                rgb = (0, 255, 0)
            else:
                rgb = (255, 0, 0)
            pygame.draw.rect(self.game_surface, rgb, (oxw - 4, oyw - 4, 8, 8))


if __name__ == "__main__":
    tsp = TSPView2D(n_orders=5, map_quad=(5, 5), grid_size=20)
    tsp.update(0, 0, 0, [1, 1, 1, 0, 0], [-3, -2, -1, 1, 2], [-3, -2, -1, 1, 2], 3, 4)
    input("Press any key to quit.")
