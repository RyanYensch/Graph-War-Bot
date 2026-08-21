import cv2
import numpy as np
import subprocess
import mss

def get_window_geometry(window_name: str):
    try:
        # Find window ID
        res = subprocess.run(
            ["xdotool", "search", "--name", window_name],
            capture_output=True, text=True, check=True
        )
        wid = res.stdout.strip().split()[0]

        # Get geometry (X, Y, W, H)
        geom = subprocess.run(
            ["xdotool", "getwindowgeometry", "--shell", wid],
            capture_output=True, text=True, check=True
        )

        props = dict(line.split('=') for line in geom.stdout.splitlines() if '=' in line)
        return int(props['X']), int(props['Y']), int(props['WIDTH']), int(props['HEIGHT'])
    except (subprocess.CalledProcessError, IndexError, KeyError):
        return None

def capture_window_by_name(window_name: str) -> np.ndarray | None:
    geom = get_window_geometry(window_name)
    if not geom:
        return None

    x, y, w, h = geom

    # Use mss to capture the specific screen region
    with mss.mss() as sct:
        monitor = {"left": x, "top": y, "width": w, "height": h}
        img = sct.grab(monitor)
        frame = np.array(img)
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

def main():
    img = capture_window_by_name("Graphwar")
    if img is not None:
        print(f"Success! Shape: {img.shape}")
        cv2.imshow("Graphwar", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Window not found.")

if __name__ == "__main__":
    main()