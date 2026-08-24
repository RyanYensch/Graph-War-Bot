import cv2
import numpy as np
import math
import random
from dataclasses import dataclass

X_MIN = -25.0
X_MAX = 25.0
Y_MIN = -15.0
Y_MAX = 15.0

EPS = 1e-9

@dataclass(frozen=True)
class Waypoint:
    point: tuple[float, float]

    enemy_index: int | None = None


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
            players.append(pixel_to_coord(img, x + w / 2, y + h / 2))

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



def attack_enemies(
    x: float,
    y: float,
    enemies: list[tuple[float, float]],
    obstacles: list[tuple[float, float, float]]
) -> tuple[
    str,
    list[tuple[float, float]],
    list[tuple[float, float]]
]:
    start = (x, y)

    path, hit_enemies = find_best_path(
        start,
        enemies,
        obstacles
    )

    equation = generate_equation_from_points(path)

    return equation, path, hit_enemies


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


def format_abs(value: float) -> str:
    if abs(value) < EPS:
        return "abs(x)"

    if value > 0:
        return f"abs(x - {value:.5g})"

    return f"abs(x + {abs(value):.5g})"


def generate_equation_from_points(
    points: list[tuple[float, float]]
) -> str:
    terms: list[str] = []

    for i in range(len(points) - 1):
        x1, y1 = points[i]
        x2, y2 = points[i + 1]

        if x2 <= x1:
            continue

        slope = (y2 - y1) / (x2 - x1)

        m = slope / 2

        if abs(m) < EPS:
            continue

        abs1 = format_abs(x1)
        abs2 = format_abs(x2)

        expression = (
            f"{abs(m):.5g}"
            f"*({abs1} - {abs2})"
        )

        if not terms:
            if m < 0:
                terms.append("-" + expression)
            else:
                terms.append(expression)

        elif m < 0:
            terms.append(" - " + expression)

        else:
            terms.append(" + " + expression)

    if not terms:
        return "0"

    return "".join(terms)


def inflate_obstacles(obstacles: list[tuple[float, float, float]], clearence: float) -> list[tuple[float, float, float]]:
    return [(x, y, r + clearence) for x, y, r in obstacles]


def point_in_bounds(point: tuple[float, float]) -> bool:
    x, y = point

    return (X_MIN <= x <= X_MAX and Y_MIN <= y <= Y_MAX)


def point_is_safe(point: tuple[float, float], obstacles: list[tuple[float, float, float]]) -> bool:
    if not point_in_bounds(point):
        return False

    x, y = point

    for cx, cy, r in obstacles:
        if (x - cx) ** 2 + (y - cy) ** 2 < r ** 2:
            return False

    return True


def generate_obstacle_detour(obstacles: list[tuple[float, float, float]]) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []

    for cx, cy, r in obstacles:
        x_positions = [cx - r, cx, cx + r]

        y_positions = [cy - r, cy + r, Y_MIN, Y_MAX]

        for x in x_positions:
            for y in y_positions:
                points.append((x, y))

    return points


def can_travel(p1: tuple[float, float], p2: tuple[float, float], obstacles: list[tuple[float, float, float]]) -> bool:
    x1, _ = p1
    x2, _ = p2

    if x2 <= x1 + EPS:
        return False

    if not  point_in_bounds(p1):
        return False

    if not point_in_bounds(p2):
        return False

    return is_line_clear(p1, p2, obstacles)


def distance_point_to_segment(
    point: tuple[float, float],
    p1: tuple[float, float],
    p2: tuple[float, float]
) -> float:
    px, py = point
    x1, y1 = p1
    x2, y2 = p2

    dx = x2 - x1
    dy = y2 - y1

    length_sq = dx * dx + dy * dy

    if length_sq == 0:
        return math.dist(point, p1)

    t = (
        (px - x1) * dx
        + (py - y1) * dy
    ) / length_sq

    t = max(0.0, min(1.0, t))

    closest_x = x1 + t * dx
    closest_y = y1 + t * dy

    return math.dist(
        point,
        (closest_x, closest_y)
    )


def enemies_hit_on_segment(
    p1: tuple[float, float],
    p2: tuple[float, float],
    enemies: list[tuple[float, float]],
    hit_tolerance: float = 0.35
) -> list[int]:
    x1, _ = p1
    x2, _ = p2

    hit: list[int] = []

    for i, enemy in enumerate(enemies):
        enemy_x, _ = enemy

        if enemy_x <= x1 + EPS:
            continue

        if enemy_x > x2 + EPS:
            continue

        distance = distance_point_to_segment(
            enemy,
            p1,
            p2
        )

        if distance <= hit_tolerance:
            hit.append(i)

    return hit



def build_waypoints(
    start: tuple[float, float],
    enemies: list[tuple[float, float]],
    obstacles: list[tuple[float, float, float]]
) -> list[Waypoint]:

    waypoints: list[Waypoint] = [
        Waypoint(start)
    ]

    if not enemies:
        return waypoints

    rightmost_enemy_x = max(
        enemy[0]
        for enemy in enemies
    )

    for i, enemy in enumerate(enemies):
        if enemy[0] <= start[0] + EPS:
            continue

        if point_is_safe(enemy, obstacles):
            waypoints.append(
                Waypoint(
                    enemy,
                    enemy_index=i
                )
            )

    used_points = {
        (
            round(w.point[0], 8),
            round(w.point[1], 8)
        )
        for w in waypoints
    }

    for point in generate_obstacle_detour(obstacles):
        x, _ = point

        if x <= start[0] + EPS:
            continue

        if x > rightmost_enemy_x + EPS:
            continue

        key = (
            round(point[0], 8),
            round(point[1], 8)
        )

        if key in used_points:
            continue

        if not point_is_safe(
            point,
            obstacles
        ):
            continue

        waypoints.append(
            Waypoint(point)
        )

        used_points.add(key)

    waypoints.sort(
        key=lambda waypoint: waypoint.point[0]
    )

    return waypoints


def find_best_path(
    start: tuple[float, float],
    enemies: list[tuple[float, float]],
    obstacles: list[tuple[float, float, float]],
    clearence: float = 0.15,
    hit_tolerance: float = 0.35
) -> tuple[
    list[tuple[float, float]],
    list[tuple[float, float]]
]:

    if not enemies:
        return [start], []

    safe_obstacles = inflate_obstacles(
        obstacles,
        clearence
    )

    waypoints = build_waypoints(
        start,
        enemies,
        safe_obstacles
    )

    start_index = next(
        i
        for i, waypoint in enumerate(waypoints)
        if waypoint.point == start
    )

    n = len(waypoints)

    hits = [-1] * n

    dist = [float("inf")] * n
    segments = [float("inf")] * n

    parent: list[int | None] = [None] * n

    hits[start_index] = 0
    dist[start_index] = 0.0
    segments[start_index] = 0

    for j in range(start_index + 1, n):
        curr = waypoints[j]

        for i in range(start_index, j):
            if hits[i] == -1:
                continue

            previous = waypoints[i]

            if not can_travel(
                previous.point,
                curr.point,
                safe_obstacles
            ):
                continue

            edge_hits = enemies_hit_on_segment(
                previous.point,
                curr.point,
                enemies,
                hit_tolerance
            )

            new_hits = (
                hits[i]
                + len(edge_hits)
            )

            new_dist = (
                dist[i]
                + math.dist(
                    previous.point,
                    curr.point
                )
            )

            new_segments = segments[i] + 1

            better = False

            if new_hits > hits[j]:
                better = True

            elif new_hits == hits[j]:

                if new_dist < dist[j] - EPS:
                    better = True

                elif (
                    abs(new_dist - dist[j]) <= EPS
                    and new_segments < segments[j]
                ):
                    better = True

            if better:
                hits[j] = new_hits
                dist[j] = new_dist
                segments[j] = new_segments
                parent[j] = i

    best_index = start_index

    for i in range(start_index, n):

        if hits[i] > hits[best_index]:
            best_index = i

        elif hits[i] == hits[best_index]:

            if dist[i] < dist[best_index]:
                best_index = i

    path: list[tuple[float, float]] = []

    current: int | None = best_index

    while current is not None:
        path.append(
            waypoints[current].point
        )

        current = parent[current]

    path.reverse()

    hit_indices: set[int] = set()

    for i in range(len(path) - 1):
        segment_hits = enemies_hit_on_segment(
            path[i],
            path[i + 1],
            enemies,
            hit_tolerance
        )

        hit_indices.update(segment_hits)

    hit_enemies = [
        enemies[i]
        for i in sorted(hit_indices)
    ]

    return path, hit_enemies



def draw_attack_path(
    img: np.ndarray,
    path: list[tuple[float, float]],
    hit_enemies: list[tuple[float, float]]
) -> None:
    for i in range(len(path) - 1):
        p1 = coord_to_pixel(
            img,
            path[i][0],
            path[i][1]
        )

        p2 = coord_to_pixel(
            img,
            path[i + 1][0],
            path[i + 1][1]
        )

        cv2.line(
            img,
            p1,
            p2,
            (0, 165, 255),
            2
        )

    for i, point in enumerate(path):
        pixel = coord_to_pixel(
            img,
            point[0],
            point[1]
        )

        if i == 0:
            colour = (0, 255, 0)
        elif i == len(path) - 1:
            colour = (255, 0, 255)
        else:
            colour = (255, 255, 0)

        cv2.circle(
            img,
            pixel,
            5,
            colour,
            -1
        )

        cv2.putText(
            img,
            str(i),
            (
                pixel[0] + 7,
                pixel[1] - 7
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            colour,
            1,
            cv2.LINE_AA
        )

    for enemy in hit_enemies:
        pixel = coord_to_pixel(
            img,
            enemy[0],
            enemy[1]
        )

        cv2.circle(
            img,
            pixel,
            10,
            (0, 0, 255),
            2
        )