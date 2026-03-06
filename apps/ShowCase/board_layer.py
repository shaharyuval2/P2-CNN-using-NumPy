import numpy as np
from board import Board
from rect import Rect


class BoardLayer:
    def __init__(
        self,
        screen,
        size: int,
        pixel_size: int,
        y: int,
        number_of_boards: int,
        pad: int,
        is_drawable=False,
    ):
        self.boards = []
        self.pixel_size = pixel_size

        # Calculate dimensions
        board_dim = size * pixel_size

        # Calculate horizontal centering
        total_width = (number_of_boards * board_dim) + ((number_of_boards - 1) * pad)

        # Start x so that the group is centered on the screen
        start_x = (screen.get_width() - total_width) // 2

        # Initialize the Boards
        for i in range(number_of_boards):
            current_x = start_x + i * (board_dim + pad)
            current_y = y - (board_dim // 2)

            rect_obj = Rect(current_x, current_y, board_dim, board_dim)
            self.boards.append(Board(rect_obj, size, is_drawable))

    def draw(self, screen, border_color=None, border_width=2):
        for board in self.boards:
            board.draw(screen, border_color=border_color, border_width=border_width)

    def draw_paint(self, screen):
        for board in self.boards:
            board.draw_paint(screen)

    def paint(self, mouse_pos):
        for board in self.boards:
            if board.is_drawable:
                if board.painting_area_rect.collidepoint(mouse_pos):
                    mouse_board_pos = mouse_pos - np.array(board.Rect.topleft)
                    pos_idx = mouse_board_pos // self.pixel_size
                    board.paint(pos_idx)

    def clear(self):
        for board in self.boards:
            board.clear()

    def set_activations(self, data: np.ndarray):
        # normalizing data
        layer_min = np.min(data)
        layer_max = np.max(data)
        denominator = layer_max - layer_min
        if denominator == 0:
            data = data - layer_min
        else:
            data = (data - layer_min) / denominator

        # Case A: Conv2d or Pooling (Shape: Channels, Height, Width)
        if data.ndim == 3:
            for i in range(len(self.boards)):
                self.boards[i].update_IMG(data[i])

        # Case B: Final Output (Shape: Neurons, 1)
        elif data.ndim == 2 or data.ndim == 1:
            flat_data = data.flatten()
            for i in range(len(self.boards)):
                # We wrap the single value in a 1x1 array so Board.draw()
                # can still iterate over it like an image
                self.boards[i].update_IMG(np.array([[flat_data[i]]]))
