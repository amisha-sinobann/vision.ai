import easyocr
import numpy as np

class OCRDetector:
    def __init__(self, languages=['en']):
        print("[OCR] Initializing EasyOCR...")
        self.reader = easyocr.Reader(languages, gpu=False) # Set gpu=True if available
        print("[OCR] Initialized.")

    def detect(self, frame):
        """
        Detect text in the frame.
        Returns a list of detections: [(bbox, text, prob), ...]
        """
        try:
            results = self.reader.readtext(frame)
            detections = []
            for (bbox, text, prob) in results:
                # bbox is list of 4 points [[x,y], [x,y], [x,y], [x,y]]
                # Convert to [x1, y1, x2, y2] for consistency
                (tl, tr, br, bl) = bbox
                x1 = int(min(tl[0], bl[0]))
                y1 = int(min(tl[1], tr[1]))
                x2 = int(max(tr[0], br[0]))
                y2 = int(max(bl[1], br[1]))
                
                detections.append({
                    "label": text,
                    "confidence": float(prob),
                    "bbox": [x1, y1, x2, y2],
                    "type": "text"
                })
            return {"text_detections": detections}
        except Exception as e:
            print(f"[OCR] Error: {e}")
            return {"text_detections": []}
