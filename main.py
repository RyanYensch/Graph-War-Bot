import cv2
from window_capture import capture_window_by_name
from graph_war_map import get_grid, label_coord, get_players

def main():
    img = capture_window_by_name("Graphwar")

    if img is not None:
        hsv_full = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        cropped = get_grid(hsv_full)

        players = get_players(cropped)

        for x, y in players:
            label_coord(cropped, x, y)
            print(x, y)

        bgr_for_display = cv2.cvtColor(cropped, cv2.COLOR_HSV2BGR)
        cv2.imshow("Cropped", bgr_for_display)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()