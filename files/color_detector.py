import cv2
import numpy as np
import colorsys


class ColorDetector:
    """Detects the dominant color(s) in a frame using HSV clustering."""

    COLOUR_NAMES = [
        ("red",    (0,   100, 100), (10,  255, 255)),
        ("red",    (160, 100,  80), (180, 255, 255)),  # red wraps around 180
        ("orange", (11,  100,  80), (25,  255, 255)),
        ("yellow", (26,  100,  80), (34,  255, 255)),
        ("green",  (35,  40,   40), (85,  255, 255)),
        ("cyan",   (86,  40,   40), (100, 255, 255)),
        ("blue",   (101, 60,   40), (130, 255, 255)),
        ("purple", (131, 60,   40), (159, 255, 255)),
        ("pink",   (160, 30,  100), (179, 130, 255)),
        ("white",  (0,   0,   200), (180,  40, 255)),
        ("grey",   (0,   0,    50), (180,  40, 200)),
        ("black",  (0,   0,    0),  (180, 255,  50)),
    ]

    def detect(self, frame):
        """Return dominant colour name and hex from the frame."""
        try:
            small = cv2.resize(frame, (160, 120))
            hsv   = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)

            best_name  = "unknown"
            best_count = 0

            for name, lo, hi in self.COLOUR_NAMES:
                mask  = cv2.inRange(hsv, np.array(lo), np.array(hi))
                count = int(np.sum(mask > 0))
                if count > best_count:
                    best_count = count
                    best_name  = name

            # Also compute mean BGR → hex
            pixels = small.reshape(-1, 3).astype(np.float32)
            mean_b, mean_g, mean_r = np.mean(pixels, axis=0)
            hex_col = "#{:02x}{:02x}{:02x}".format(
                int(mean_r), int(mean_g), int(mean_b)
            )

            return {"dominant_color": best_name, "hex": hex_col}
        except Exception as e:
            return {"dominant_color": "unknown", "hex": "#000000"}
