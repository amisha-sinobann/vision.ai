import cv2
import numpy as np

# ── Indian Rupee note colour signatures (HSV dominant hue ranges) ──────────
# Each entry: (denomination_label, lower_hsv, upper_hsv, fallback_colour_name)
_RUPEE_SIGNATURES = [
    ("₹2000", ( 140, 60,  60), (175, 255, 255), "magenta/pink"),
    ("₹500",  (  0,  0,  90), ( 20,  40, 210), "stone grey"),
    ("₹200",  ( 22, 80,  80), ( 35, 255, 255), "yellow"),
    ("₹100",  (120, 40,  60), (145, 255, 255), "lavender/blue"),
    ("₹50",   (  8, 80,  80), ( 22, 255, 255), "fluorescent blue"),  # blue-green edge
    ("₹20",   ( 36, 60,  60), ( 75, 255, 255), "green-yellow"),
    ("₹10",   ( 10, 60,  40), ( 20, 200, 180), "chocolate/brown"),
]

# Minimum fraction of pixels that must match the signature
_MIN_COVERAGE = 0.08


class CurrencyDetector:
    """Detect Indian Rupee notes by dominant colour and OCR number recognition."""

    def __init__(self, conf_threshold=0.5):
        self.conf_threshold = conf_threshold
        # Try to import EasyOCR for number confirmation
        try:
            import easyocr
            self._reader = easyocr.Reader(["en"], gpu=False, verbose=False)
        except Exception:
            self._reader = None

    # ------------------------------------------------------------------
    def detect(self, frame, crop=None):
        """
        Analyse *frame* (or the *crop* region if given) for a currency note.
        Returns a dict with keys: detected, label, denomination, confidence.
        """
        if frame is None:
            return {"detected": False}

        # Crop to the hint region if provided
        roi = frame
        if crop and len(crop) == 4:
            x, y, w, h = crop
            fh, fw = frame.shape[:2]
            # Ensure ROI is within frame bounds
            x1, y1 = max(0, int(x)), max(0, int(y))
            x2, y2 = min(fw, int(x + w)), min(fh, int(y + h))
            if x2 > x1 and y2 > y1:
                roi = frame[y1:y2, x1:x2]

        try:
            # Resize for faster processing
            small = cv2.resize(roi, (320, 240))
            
            # 1. OCR Check (Primary for specific symbol/value)
            ocr_text = ""
            ocr_denom = None
            if hasattr(self, '_reader') and self._reader:
                try:
                    # EasyOCR expects RGB, OpenCV gives BGR
                    rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                    results = self._reader.readtext(rgb_small, detail=0)
                    ocr_text = " ".join([r.lower() for r in results])
                    
                    # Check for Rupee symbol or specific keywords
                    for denom_str, _, _, _ in _RUPEE_SIGNATURES:
                        # Extract number (e.g. "500" from "₹500")
                        val = denom_str.replace("₹", "")
                        if val in ocr_text or "rupee" in ocr_text or "₹" in ocr_text:
                             if val in ocr_text:
                                 ocr_denom = denom_str
                                 break
                except Exception as e:
                    print(f"OCR Error: {e}")

            # 2. Color Analysis (Secondary / Validation)
            hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
            total_pixels = small.shape[0] * small.shape[1]
            best_denom = None
            max_coverage = 0.0

            for (denom, (lo0,lo1,lo2), (hi0,hi1,hi2), _) in _RUPEE_SIGNATURES:
                lower = np.array([lo0, lo1, lo2], dtype="uint8")
                upper = np.array([hi0, hi1, hi2], dtype="uint8")
                mask = cv2.inRange(hsv, lower, upper)
                coverage = np.count_nonzero(mask) / total_pixels
                
                if coverage > max_coverage:
                    max_coverage = coverage
                    best_denom = denom

            # Decision Logic
            final_denom = None
            confidence = 0.0
            
            # Strong match if OCR captures the number
            if ocr_denom:
                final_denom = ocr_denom
                confidence = 0.95
            
            # REMOVED: purely color-based detection to prevent false positives.
            # We now require OCR confirmation (symbol/value) as requested.
            # elif max_coverage > 0.15: ...

            if final_denom and confidence > self.conf_threshold:
                return {
                    "detected": True,
                    "label": f"Indian Rupee {final_denom}",
                    "likely_currency": f"{final_denom} Rupee Note",
                    "denomination": final_denom,
                    "confidence": confidence,
                    "ocr_text": ocr_text
                }

            return {"detected": False}

        except Exception as e:
            print(f"Currency Detector Error: {e}")
            return {"detected": False}
