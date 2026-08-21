import cv2
import numpy as np
import subprocess
import mss

def get_window_geometry(window_name: str):
    try:
        res = subprocess.run(
            ["xdotool", "search", "--onlyvisible", "--name", window_name],
            capture_output=True, text=True, check=True
        )

        windows = res.stdout.strip().split()
        if not windows:
            return None

        active_win = subprocess.run(
            ["xdotool", "getactivewindow"],
            capture_output=True, text=True, check=True
        ).stdout.strip()

        target_wid = None
        for wid in windows:
            if wid == active_win:
                target_wid = wid
                break

        if not target_wid:
            target_wid = windows[-1]

        geom = subprocess.run(
            ["xdotool", "getwindowgeometry", "--shell", target_wid],
            capture_output=True, text=True, check=True
        )

        props = dict(line.split('=') for line in geom.stdout.splitlines() if '=' in line)

        x, y, w, h = int(props['X']), int(props['Y']), int(props['WIDTH']), int(props['HEIGHT'])

        return x, y, w, h

    except (subprocess.CalledProcessError, IndexError, KeyError, ValueError) as e:
        print(f"Error finding window: {e}")
        return None

def capture_window_by_name(window_name: str) -> np.ndarray | None:
    geom = get_window_geometry(window_name)
    if not geom:
        return None

    x, y, w, h = geom
    print(f"Capturing region: x={x}, y={y}, w={w}, h={h}")

    with mss.MSS() as sct:  # Use mss.MSS() to fix deprecation warning
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
        print("Window not found or invalid geometry.")

if __name__ == "__main__":
    main()