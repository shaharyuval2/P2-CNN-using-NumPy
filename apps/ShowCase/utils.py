import numpy as np

COLOURS = {
    "main": (31, 31, 31),
    "secondary": (20, 20, 20),
    "third": (88, 146, 83),
    "text": (244, 246, 251),
    "black": (0, 0, 0),
    "white": (255, 255, 255),
}


def lerp_color(colorA, colorB, p):
    return (
        colorA[0] * (1 - p) + colorB[0] * p,
        colorA[1] * (1 - p) + colorB[1] * p,
        colorA[2] * (1 - p) + colorB[2] * p,
    )


def create_kernel(center, side, corner):
    return np.array(
        [[corner, side, corner], [side, center, side], [corner, side, corner]]
    )


def center_image(img):
    # Get the coordinates of all non-zero pixels
    rows, cols = np.where(img > 0)
    if len(rows) == 0:
        return img  # Image is empty

    # Calculate center of mass
    weights = img[rows, cols]
    m_y = np.average(rows, weights=weights)
    m_x = np.average(cols, weights=weights)

    # Calculate the shift required to get to (14, 14)
    shift_y = int(round(14 - m_y))
    shift_x = int(round(14 - m_x))

    # Use np.roll to shift the image
    centered_img = np.roll(img, shift_y, axis=0)
    centered_img = np.roll(centered_img, shift_x, axis=1)

    # Clean up "wrap-around" edges
    # np.roll wraps pixels from the right side to the left.
    # For a centered digit, we want those gaps to be 0 (black).
    if shift_y > 0:
        centered_img[:shift_y, :] = 0
    elif shift_y < 0:
        centered_img[shift_y:, :] = 0

    if shift_x > 0:
        centered_img[:, :shift_x] = 0
    elif shift_x < 0:
        centered_img[:, shift_x:] = 0

    return centered_img
