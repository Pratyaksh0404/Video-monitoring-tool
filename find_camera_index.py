"""
find_camera_index.py
────────────────────────
Opens each camera device index (0-4) in turn, shows a preview window, and
tells you which index is which physical camera. Windows/OpenCV doesn't
guarantee the built-in laptop webcam is always index 0 — it depends on
driver load order — so this checks visually instead of guessing.

Usage:
    python find_camera_index.py

Press any key to move to the next index. Press 'q' to quit early once
you've found the one you need.
"""

import cv2

for index in range(5):
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)  # DSHOW backend is more reliable on Windows
    if not cap.isOpened():
        print(f"Index {index}: no camera found here")
        cap.release()
        continue

    ret, frame = cap.read()
    if not ret or frame is None:
        print(f"Index {index}: opened but no frame — probably not a real camera")
        cap.release()
        continue

    print(f"Index {index}: camera found — showing preview window. "
          f"Press any key for next index, or 'q' to stop.")
    cv2.imshow(f"Camera index {index} — press any key for next", frame)
    key = cv2.waitKey(0)
    cv2.destroyAllWindows()
    cap.release()

    if key == ord('q'):
        break

print("\nDone. Use the index number shown on the preview window that")
print("matched your USB camera as the 'source:' value in camera_config.yaml.")
