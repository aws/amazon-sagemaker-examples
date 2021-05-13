import numpy as np
import pygame


class KnapsackView2D:
    def __init__(self, bag_weight_capacity, max_item_value, bag_volume_capacity=None):

        self.game_over = False
        self.bag_volume_capacity = bag_volume_capacity
        self.bag_weight_capacity = bag_weight_capacity
        self.max_item_value = max_item_value

        self.screen_size = (900, 500)
        pygame.init()
        pygame.font.init()
        pygame.display.set_caption("Knapsack")
        self.screen = pygame.display.set_mode(self.screen_size)

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

    def reset_game(self):
        self.game_over = False

    def update(
        self,
        selected_item_queue,
        reward,
        item,
        bag_weight,
        bag_value,
        bag_volume=None,
        mode="human",
    ):
        try:
            img = self.__draw_game(
                mode, selected_item_queue, reward, item, bag_weight, bag_volume, bag_value
            )
            self.__handle_pygame_events()
        except Exception as e:
            self.game_over = True
            pygame.display.quit()
            pygame.quit()
            raise e
        else:
            return img

    def __draw_game(
        self, mode, selected_item_queue, reward, item, bag_weight, bag_volume, bag_value
    ):

        self.game_surface.fill(
            (
                0,
                0,
                0,
                0,
            )
        )

        self.__draw_items_in_bag(selected_item_queue)
        self.__draw_next_item(item)

        # update the screen
        self.screen.blit(self.background, (0, 0))
        self.screen.blit(self.game_surface, (0, 0))
        self.screen.blit(self._create_state_surface(bag_weight, bag_volume, bag_value), (200, 450))
        self.screen.blit(
            self._create_text_surface(f"Reward: {reward}", color=(0, 255, 0)), (25, 450)
        )
        self.screen.blit(self._create_text_surface(f"Item Weight: {item.weight}"), (175, 225))
        self.screen.blit(self._create_text_surface(f"Item Value: {item.value}"), (175, 410))
        weight_capacity_remaining = self.bag_weight_capacity - bag_weight
        self.screen.blit(
            self._create_text_surface(
                f"<--- Weight- Capacity: {self.bag_weight_capacity}"
                f" Remaining: {weight_capacity_remaining} --->"
            ),
            (500, 225),
        )
        if self.bag_volume_capacity:
            volume_capacity_remaining = self.bag_volume_capacity - bag_volume
            self.screen.blit(
                self._create_text_surface(
                    f"<--- Volume- Capacity: {self.bag_volume_capacity}"
                    f" Remaining: {volume_capacity_remaining} --->"
                ),
                (500, 25),
            )
            self.screen.blit(self._create_text_surface(f"Item Volume: {item.volume}"), (175, 25))

        if mode == "human":
            pygame.display.flip()

        return np.flipud(np.rot90(pygame.surfarray.array3d(pygame.display.get_surface())))

    def __handle_pygame_events(self):
        if not self.game_over:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.game_over = True
                    pygame.display.quit()
                    pygame.quit()

    def __draw_items_in_bag(self, selected_item_queue):

        # Draw weight dimension
        end_x = 875
        end_y = 250
        start_x = 475
        start_y = 250

        pygame.draw.line(self.game_surface, (250, 50, 50), (start_x, start_y), (end_x, end_y), 4)

        for item in list(selected_item_queue.queue):
            width = self.__rescale(item.weight, self.bag_weight_capacity, 400)
            height = self.__rescale(item.value, self.max_item_value, 150)
            pygame.draw.rect(
                self.game_surface,
                self.__get_color_for_item(item),
                (end_x - width, end_y, width, height),
            )
            end_x = end_x - width

        # Draw volume dimension
        if self.bag_volume_capacity:
            end_x = 875
            end_y = 50
            start_x = 475
            start_y = 50

            pygame.draw.line(
                self.game_surface, (250, 50, 50), (start_x, start_y), (end_x, end_y), 4
            )

            for item in list(selected_item_queue.queue):
                width = self.__rescale(item.volume, self.bag_volume_capacity, 400)
                height = self.__rescale(item.value, self.max_item_value, 150)
                pygame.draw.rect(
                    self.game_surface,
                    self.__get_color_for_item(item),
                    (end_x - width, end_y, width, height),
                )
                end_x = end_x - width

    @staticmethod
    def __get_color_for_item(item):
        r = 255 - item.weight % 255
        g = 255 - item.value % 255
        b = item.volume == 0 if 125 else 255 - item.value % 255
        return r, g, b

    @staticmethod
    def _create_text_surface(text, color=(255, 0, 0)):
        font = pygame.font.SysFont("arial", 22)
        return font.render(text, True, color)

    @staticmethod
    def _create_state_surface(bag_weight, bag_volume, bag_value):
        font = pygame.font.SysFont("arial", 22)
        if bag_volume:
            return font.render(
                f"Bag-Weight: {bag_weight}    Bag-Volume: {bag_volume}"
                f"   Bag-Value: {bag_value}",
                True,
                (0, 0, 255),
            )
        else:
            return font.render(
                f"Bag-Weight: {bag_weight}" f"   Bag-Value: {bag_value}", True, (0, 0, 255)
            )

    def __draw_next_item(self, item):
        # Draw weight dimension
        end_x = 425
        end_y = 250
        width = self.__rescale(item.weight, self.bag_weight_capacity, 400)
        height = self.__rescale(item.value, self.max_item_value, 150)
        pygame.draw.rect(
            self.game_surface,
            self.__get_color_for_item(item),
            (end_x - width, end_y, width, height),
        )

        if self.bag_volume_capacity:
            # Draw volume dimension
            end_x = 425
            end_y = 50
            width = self.__rescale(item.volume, self.bag_volume_capacity, 400)
            height = self.__rescale(item.value, self.max_item_value, 150)
            pygame.draw.rect(
                self.game_surface,
                self.__get_color_for_item(item),
                (end_x - width, end_y, width, height),
            )

    @staticmethod
    def __rescale(value, max_value, new_max):
        return (value * new_max) / max_value
