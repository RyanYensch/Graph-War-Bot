import cv2
from window_capture import capture_window_by_name

def main():
    img = capture_window_by_name("Graphwar")

    if img is not None:
        print(f"Success! Shape: {img.shape}")

        cv2.imshow("Graphwar", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Window not found or invalid geometry.")

if __name__ == "__main__":
    main()