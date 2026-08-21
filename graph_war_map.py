import cv2
import numpy as np


def get_grid(graphwar_window_img: np.ndarray) -> np.ndarray:
    h, w, _ = graphwar_window_img.shape

    l, r = int(0.02 * w), int(0.98 * w)
    t, b = int(0.025 * h), int(0.77 * h)
    return graphwar_window_img[t:b, l:r]


def coord_to_pixel(img: np.ndarray, xCoord: float, yCoord: float) -> tuple[int, int]:
    h, w, _ = img.shape
    WIDTH = 50
    HEIGHT = 30

    return int(w / 2 + xCoord * w / WIDTH), int(h / 2 - yCoord * h / HEIGHT)

def pixel_to_coord(img: np.ndarray, xPixel: int, yPixel: int) -> tuple[float, float]:
    h, w, _ = img.shape
    WIDTH = 50
    HEIGHT = 30

    return xPixel * WIDTH / w - WIDTH / 2, HEIGHT / 2 - yPixel * HEIGHT / h

def y_coord_to_pixel(img: np.ndarray, yCoord: float) -> int:
    h, _, _ = img.shape
    HEIGHT = 30
    return int(h / 2 - yCoord * h / HEIGHT)

def label_coord(cropped_window: np.ndarray, xCoord: float, yCoord: float) -> None:
    x, y = coord_to_pixel(cropped_window, xCoord, yCoord)

    cv2.circle(cropped_window, (x, y), 5, (50, 255, 255), -1)

# Label and get coords
def get_players(img: np.ndarray) -> list[tuple[float, float]]:
    players: list[tuple[float, float]] = []

    hsv = img

    lower_yellow = np.array([30, 200, 200])
    upper_yellow = np.array([40, 255, 255])
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]

        if area > 50:
            players.append(pixel_to_coord(img, x + w / 2, y + w / 2))

    return players
