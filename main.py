import cv2
from window_capture import capture_window_by_name
from graph_war_map import get_grid, label_coord, label_players

def main():
    img = capture_window_by_name("Graphwar")

    if img is not None:
        hsv_full = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        cropped = get_grid(hsv_full)

        label_coord(cropped, 12.5, 7.5)
        label_players(cropped)

        bgr_for_display = cv2.cvtColor(cropped, cv2.COLOR_HSV2BGR)
        cv2.imshow("Image", img)
        cv2.imshow("Cropped", bgr_for_display)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()