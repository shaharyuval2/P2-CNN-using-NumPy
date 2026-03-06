import numpy as np
import pygame
from rect import Rect
from scipy import signal
from utils import center_image, create_kernel


class Board:
    def __init__(self, Rect: Rect, board_size: int, is_drawable=False):
        self.Rect = Rect
        self.size = board_size
        self.board_size = board_size
        self.IMG = np.zeros(shape=(board_size, board_size))
        self.IMG_smooth = np.zeros(shape=(board_size, board_size))
        self.IMG_final = np.zeros(shape=(board_size, board_size))
        self.is_drawable = is_drawable

        self.kernel = None
        self.painting_area_rect = None
        if is_drawable:
            self.kernel = create_kernel(center=1, side=0.4, corner=0.1)

            pixel_unit = self.Rect.w / self.board_size
            margin_size = 4 * pixel_unit
            self.painting_area_rect = Rect.get_padded(-margin_size)

    def draw(self, screen, border_color=None, border_width=2):
        pygame.draw.rect(
            screen,
            "black",
            self.Rect.rect,
        )

        if border_color:
            pygame.draw.rect(
                screen,
                border_color,
                self.Rect.get_padded(border_width),
                width=border_width,
                border_radius=2,
            )

        self.draw_painting_border(screen)

    def draw_painting_border(self, screen):
        if self.is_drawable:
            pygame.draw.rect(
                screen,
                (255, 0, 0),
                self.painting_area_rect.inflate(2, 2),
                width=2,
            )

    def draw_paint(self, screen):
        IMG_final = None
        if self.is_drawable:
            IMG_final = self.IMG_smooth
        else:
            IMG_final = self.IMG

        gray_values = (IMG_final * 255).astype(np.uint8)
        rgb_IMG = np.stack((gray_values,) * 3, axis=-1)
        rgb_IMG = np.transpose(rgb_IMG, (1, 0, 2))

        temp_surf = pygame.surfarray.make_surface(rgb_IMG)
        final_size = (self.Rect.w, self.Rect.h)
        scaled_surf = pygame.transform.scale(temp_surf, final_size)
        screen.blit(scaled_surf, self.Rect.topleft)

        self.draw_painting_border(screen)

    def paint(self, pos_idx):
        self.IMG[pos_idx[1], pos_idx[0]] = 1
        self.update_IMG_smooth()
        self.update_IMG_final()

    def update_IMG_smooth(self):
        self.IMG_smooth = signal.convolve2d(self.IMG, self.kernel, mode="same")
        np.clip(self.IMG_smooth, 0, 1, out=self.IMG_smooth)

    def update_IMG_final(self):
        self.IMG_final = center_image(self.IMG_smooth)

    def clear(self):
        self.IMG = np.zeros(shape=(self.board_size, self.board_size))
        self.IMG_smooth = np.zeros(shape=(self.board_size, self.board_size))
        self.IMG_final = np.zeros(shape=(self.board_size, self.board_size))

    def update_IMG(self, data: np.ndarray):
        self.IMG = data
