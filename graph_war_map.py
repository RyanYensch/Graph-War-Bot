import cv2
import numpy as np
import math
import random

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


def get_player_turn(img: np.ndarray, players: list[tuple[float, float]]) -> tuple[float, float]:

    hsv = img

    lower_red = np.array([0, 150, 150])
    upper_red = np.array([4, 255, 255])

    mask = cv2.inRange(hsv, lower_red, upper_red)


    kernel = np.ones((3,3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=6)
    mask = cv2.erode(mask, kernel, iterations=4)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

    max_area = 0
    best_x, best_y = 0, 0

    for i in range(1, num_labels):
            x, y, w, h, area = stats[i]

            if area > max_area:
                max_area = area
                best_x, best_y = pixel_to_coord(img, x + w / 2, y + h / 2)

    closest_x, closest_y = 0, 0
    closest_dist = float('inf')

    for x, y in players:
        dist = math.dist((x,y), (best_x, best_y))

        if dist < closest_dist:
            closest_dist = dist
            closest_x, closest_y = x, y

    return closest_x, closest_y



def attack_enemy(x: float, y: float, enemies: list[tuple[float, float]], obstacles: list[tuple[float, float, float]]) -> str:
    for enemy in enemies:
        if is_line_clear((x, y), enemy, obstacles):
            m = (enemy[1] - y) / (enemy[0] - x)
            return f"{m}x"

    m = (enemies[0][1] - y) / (enemies[0][0] - x)
    return f"{m}x"


# x, y, radius
def get_obstacles(img: np.ndarray) -> list[tuple[float, float, float]]:
    obstacles = []
    hsv = img

    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 50]) # Adjust '50' based on lighting/shadows

    mask = cv2.inRange(hsv, lower_black, upper_black)
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]

        if area > 100:
            radius_pixels = math.sqrt(area / math.pi)
            center_x, center_y = pixel_to_coord(img, x + w/2, y + h/2)

            right_edge_x, _ = pixel_to_coord(img, x + w/2 + radius_pixels, y + h/2)
            radius_game = abs(right_edge_x - center_x)

            obstacles.append((center_x, center_y, radius_game))
    return obstacles

def is_line_clear(p1: tuple[float, float], p2: tuple[float, float], obstacles: list[tuple[float, float, float]]) -> bool:
    x1, y1 = p1
    x2, y2 = p2

    dx, dy = x2 - x1, y2 - y1
    length_sq = dx*dx + dy*dy

    if length_sq == 0:
        return True

    for cx, cy, r in obstacles:
        cx_x1, cy_y1 = cx - x1, cy - y1
        t = (cx_x1 * dx + cy_y1 * dy) / length_sq

        t = max(0, min(1, t))
        closest_x = x1 + t * dx
        closest_y = y1 + t * dy

        dist_sq = (cx - closest_x)**2 + (cy - closest_y)**2

        if dist_sq < r * r:
            return False

    return True