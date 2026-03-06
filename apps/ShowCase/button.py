import pygame
from rect import Rect
from utils import COLOURS


class Button:
    def __init__(
        self,
        x,
        y,
        w,
        h,
        text,
        callback,
        color,
        text_color=COLOURS["text"],
        font_size=24,
    ):
        self.ui_rect = Rect(x, y, w, h)
        self.text = text
        self.callback = callback
        self.color = color

        # Setup Font
        self.font = pygame.font.SysFont("Arial", font_size)
        self.text_surf = self.font.render(self.text, True, text_color)

    def draw(self, screen, border_radius=0, shadow=False, border_color=None):
        # draw rect
        self.ui_rect.draw(screen, self.color, border_radius, shadow, border_color)

        # draw text
        text_rect = self.text_surf.get_rect(center=self.ui_rect.center)
        screen.blit(self.text_surf, text_rect)

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                if self.ui_rect.collidepoint(event.pos):
                    self.callback()
