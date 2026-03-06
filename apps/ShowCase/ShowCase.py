import numpy as np
import pygame
from board_layer import BoardLayer
from button import Button
from text_box import TextBox
from utils import COLOURS

from p2_cnn.layers.conv2d import Conv2d
from p2_cnn.layers.pooling import Pooling
from p2_cnn.model import Model

# screen and board global variables
WIDTH, HEIGHT = 1400, 900
PIXEL_SIZE = 8

# Load the model
my_model = Model.load("models/mnist_cnn_v3.npz")
guess = 0


def paint():
    # left mouse button down
    if pygame.mouse.get_pressed()[0]:
        mouse_pos = np.array(pygame.mouse.get_pos())
        for layer in layers:
            layer.paint(mouse_pos)


def clear():
    for layer in layers:
        layer.clear()


def update_boards():
    global guess
    input_data = layers[0].boards[0].IMG_final

    # tracks which layer we are updating
    display_idx = 1

    current_input = input_data.reshape(1, 28, 28)
    for i, model_layer in enumerate(my_model.layers):
        # forward for current layer
        current_input = model_layer.forward(current_input)

        # Check if it's a layer we actually want to show
        is_conv_pool = isinstance(model_layer, (Conv2d, Pooling))
        is_last_layer = i == len(my_model.layers) - 1

        # calculate the guess
        if is_last_layer:
            guess = int(np.argmax(current_input))

        # if so, update board_layer accordingly
        if is_conv_pool or is_last_layer:
            if display_idx < len(layers):
                layers[display_idx].set_activations(current_input)
                display_idx += 1


pygame.init()
screen = pygame.display.set_mode([WIDTH, HEIGHT])

fps = 60
clock = pygame.time.Clock()
pygame.display.set_caption("My Neural Net Showcase!")

screen.fill((100, 100, 100))

# define clear button
clear_button = Button(
    900, 100, 100, 60, "Clear", clear, COLOURS["secondary"], font_size=30
)
output_values_textbox = TextBox(
    500,
    770,
    400,
    40,
    "0         1         2         3         4         5         6         7         8         9",
    font_size=18,
)
machine_guess_textbox = TextBox(
    550,
    820,
    300,
    50,
    f"Machine guess: {guess}",
    font_size=30,
    bg_color=COLOURS["secondary"],
)
# define board layers
layers = []
layers.append(BoardLayer(screen, 28, PIXEL_SIZE, 130, 1, 60, is_drawable=True))
layers.append(BoardLayer(screen, 24, PIXEL_SIZE, 360, 6, 20))
layers.append(BoardLayer(screen, 12, PIXEL_SIZE, 520, 6, 116))
layers.append(BoardLayer(screen, 8, PIXEL_SIZE, 625, 12, 42))
layers.append(BoardLayer(screen, 4, PIXEL_SIZE, 695, 12, 74))
layers.append(BoardLayer(screen, 1, 2 * PIXEL_SIZE, 760, 10, 30))


# ----- Main Loop -----
run = True
while run:
    clock.tick(fps)
    # clear screen
    screen.fill(COLOURS["main"])

    # draw all of the elements
    for layer in layers:
        layer.draw(screen, border_color=COLOURS["white"], border_width=2)
    for layer in layers:
        layer.draw_paint(screen)
    clear_button.draw(
        screen, border_radius=6, shadow=True, border_color=COLOURS["white"]
    )
    output_values_textbox.draw(screen)
    machine_guess_textbox.draw(
        screen, border_radius=10, shadow=True, border_color=COLOURS["white"]
    )

    paint()
    update_boards()
    machine_guess_textbox.update_text(f"Machine guess: {guess}")

    # handle events
    for event in pygame.event.get():
        # window closed event
        if event.type == pygame.QUIT:
            run = False
        clear_button.handle_event(event)

    pygame.display.flip()

pygame.quit()
