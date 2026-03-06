import pygame
from rect import Rect
from utils import COLOURS


class TextBox:
    def __init__(
        self, x, y, w, h, text, font_size=24, color=COLOURS["text"], bg_color=None
    ):
        self.ui_rect = Rect(x, y, w, h)
        self.text = text
        self.color = color
        self.bg_color = bg_color

        self.font = pygame.font.SysFont("Arial", font_size)
        self._render_text()

    def _render_text(self):
        """Pre-renders the text surface for performance."""
        self.text_surf = self.font.render(str(self.text), True, self.color)

    def update_text(self, new_text):
        if str(new_text) != str(self.text):
            self.text = new_text
            self._render_text()

    def draw(self, screen, border_radius=0, shadow=False, border_color=None):
        # draw background
        if self.bg_color:
            self.ui_rect.draw(
                screen,
                self.bg_color,
                border_radius=border_radius,
                shadow=shadow,
                border_color=border_color,
            )

        # draw text
        text_rect = self.text_surf.get_rect(center=self.ui_rect.center)
        screen.blit(self.text_surf, text_rect)
