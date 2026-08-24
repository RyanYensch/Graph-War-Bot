import cv2
from window_capture import capture_window_by_name
from graph_war_map import (
    get_grid,
    label_coord,
    get_players,
    get_player_turn,
    attack_enemies,
    get_obstacles,
    draw_attack_path,
)


def main():
    img = capture_window_by_name("Graphwar")

    if img is None:
        print("Could not capture Graphwar window")
        return

    # Convert to HSV for detection
    hsv_full = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    cropped = get_grid(hsv_full)

    obstacles = get_obstacles(cropped)
    players = get_players(cropped)

    team: list[tuple[float, float]] = []
    enemies: list[tuple[float, float]] = []

    for player in players:
        if player[0] < 0:
            team.append(player)
        else:
            enemies.append(player)

    curr_x, curr_y = get_player_turn(cropped, players)

    # Convert back to BGR BEFORE drawing anything.
    bgr_for_display = cv2.cvtColor(
        cropped,
        cv2.COLOR_HSV2BGR
    )

    # Mark the current player.
    label_coord(
        bgr_for_display,
        curr_x,
        curr_y
    )

    if (curr_x, curr_y) in team:
        if not enemies:
            print("No enemies detected")

        else:
            equation, path, hit_enemies = attack_enemies(
                curr_x,
                curr_y,
                enemies,
                obstacles
            )

            print(f"Current player: ({curr_x:.2f}, {curr_y:.2f})")

            print("\nPath:")
            for i, point in enumerate(path):
                print(
                    f"  {i}: "
                    f"({point[0]:.2f}, {point[1]:.2f})"
                )

            print(
                f"\nEnemies hit: "
                f"{len(hit_enemies)}/{len(enemies)}"
            )

            for enemy in hit_enemies:
                print(
                    f"  ({enemy[0]:.2f}, {enemy[1]:.2f})"
                )

            print("\nEquation:")
            print(equation)

            draw_attack_path(
                bgr_for_display,
                path,
                hit_enemies
            )

    else:
        print("It is currently the enemy team's turn")

    cv2.imshow(
        "Graphwar Attack Path",
        bgr_for_display
    )

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()