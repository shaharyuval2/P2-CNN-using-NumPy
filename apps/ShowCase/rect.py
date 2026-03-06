import pygame
from utils import lerp_color


class Rect:
    def __init__(self, x: int, y: int, w: int, h: int):
        self.rect = pygame.Rect(x, y, w, h)

    def __getattr__(self, name):
        # If someone asks for my_obj.center, and it's not in our class,
        # we check if it exists in our internal self.rect
        return getattr(self.rect, name)

    def __setattr__(self, name, value):
        # We need to handle initialization specifically to avoid recursion
        if name == "rect":
            super().__setattr__(name, value)
        else:
            setattr(self.rect, name, value)

    @staticmethod
    def _parse_padding(padding):
        if isinstance(padding, (int, float)):
            return (padding, padding)
        return padding

    def get_padded(self, padding):
        px, py = self._parse_padding(padding)
        return self.rect.inflate(px * 2, py * 2)

    def draw(self, screen, color, border_radius=0, shadow=False, border_color=None):
        # Draw Shadow (Offset to the bottom-right)
        if shadow:
            shadow_rect = self.rect.move(7, 7)
            shadow_color = lerp_color(color, (0, 0, 0), 0.8)
            pygame.draw.rect(
                screen, shadow_color, shadow_rect, border_radius=border_radius
            )

        # Draw Main Rect
        pygame.draw.rect(screen, color, self.rect, border_radius=border_radius)

        # Draw Border
        if border_color:
            pygame.draw.rect(
                screen,
                border_color,
                self.get_padded(2),
                width=1,
                border_radius=border_radius,
            )
