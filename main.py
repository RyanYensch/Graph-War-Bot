import cv2
from window_capture import capture_window_by_name
from graph_war_map import get_grid, label_coord, get_players, get_player_turn, attack_enemy, get_obstacles, is_line_clear
import random

def main():
    img = capture_window_by_name("Graphwar")

    if img is not None:
        hsv_full = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        cropped = get_grid(hsv_full)

        obstacles = get_obstacles(cropped)

        players = get_players(cropped)

        team: list[tuple[float, float]] = []
        enemy: list[tuple[float, float]] = []

        for player in players:
            if player[0] < 0:
                team.append(player)
            else:
                enemy.append(player)

        curr_x, curr_y = get_player_turn(cropped, players)
        label_coord(cropped, curr_x, curr_y)

        if (curr_x, curr_y) in team:
            print(attack_enemy(curr_x, curr_y, enemy, obstacles))

        bgr_for_display = cv2.cvtColor(cropped, cv2.COLOR_HSV2BGR)
        cv2.imshow("Cropped", bgr_for_display)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()