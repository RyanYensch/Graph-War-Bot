import cv2
import numpy as np


def get_grid(graphwar_window_img: np.ndarray) -> np.ndarray:
    h, w, _ = graphwar_window_img.shape

    l, r = int(0.02 * w), int(0.98 * w)
    t, b = int(0.025 * h), int(0.77 * h)
    return graphwar_window_img[t:b, l:r]


def label_coord(cropped_window: np.ndarray, xCoord: float, yCoord: float) -> None:
    h, w, _ = cropped_window.shape

    WIDTH = 50
    HEIGHT = 30

    x = int(w / 2 + xCoord * w / WIDTH)
    y = int(h / 2 - yCoord * h / HEIGHT)

    cv2.circle(cropped_window, (x, y), 5, (0, 255, 0), -1)