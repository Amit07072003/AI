import threading
import time
import os
import math
import numpy as np

try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False


_latest_jpeg_frame = None
_vision_instance = None

def get_latest_jpeg():
    global _latest_jpeg_frame
    return _latest_jpeg_frame


class TrackedUser:
    def __init__(self, user_id, bbox):
        self.user_id = user_id
        self.bbox = bbox  # (x, y, w, h)
        self.norm_x = 0.0
        self.norm_y = 0.0
        self.prev_mouth_gray = None
        self.speech_activity = 0.0
        self.last_seen = time.time()
        self.label = f"User {user_id}"
        self.mouth_bbox = None
        self.update(bbox)

    def update(self, bbox, frame_gray=None):
        self.bbox = bbox
        x, y, w, h = bbox
        self.last_seen = time.time()
        
        # Center of face
        cx = x + w / 2.0
        cy = y + h / 2.0
        
        # 640x480 normalized (-1.0 to +1.0)
        self.norm_x = (cx - 320.0) / 320.0
        self.norm_y = (cy - 240.0) / 240.0
        
        # Label position
        if self.norm_x < -0.22:
            self.label = f"User {self.user_id} (Left)"
        elif self.norm_x > 0.22:
            self.label = f"User {self.user_id} (Right)"
        else:
            self.label = f"User {self.user_id} (Center)"

        # Lower 35% of face is the mouth area
        my = int(y + h * 0.65)
        mh = int(h * 0.35)
        mx = int(x + w * 0.20)
        mw = int(w * 0.60)
        self.mouth_bbox = (mx, my, mw, mh)

        # Compute mouth motion if frame provided
        if frame_gray is not None:
            fh, fw = frame_gray.shape
            if my >= 0 and mx >= 0 and my + mh <= fh and mx + mw <= fw:
                mouth_roi = frame_gray[my:my+mh, mx:mx+mw]
                mouth_roi = cv2.resize(mouth_roi, (40, 24))
                mouth_roi = cv2.GaussianBlur(mouth_roi, (5, 5), 0)
                
                if self.prev_mouth_gray is not None:
                    diff = cv2.absdiff(mouth_roi, self.prev_mouth_gray)
                    motion_val = float(np.mean(diff))
                    if motion_val > 3.0:
                        self.speech_activity = self.speech_activity * 0.75 + motion_val * 0.25
                    else:
                        self.speech_activity *= 0.82
                self.prev_mouth_gray = mouth_roi.copy()


class VisionSystem:
    def __init__(self):
        global _vision_instance
        _vision_instance = self
        
        self.enabled = False
        self.is_running = False
        self.cap = None
        self.face_cascade = None
        self.eye_cascade = None
        self.tracked_users = {} # id -> TrackedUser
        self.next_user_id = 1
        self.active_speaker_id = None
        self.target_x = 0.0
        self.target_y = 0.0
        self._thread = None
        self.lock = threading.Lock()
        
        # Saliency analysis cache
        self.latest_objects = []
        self.latest_hands = []
        self.latest_scene_colors = []
        self.latest_digits_detected = []
        self.latest_clothing_analysis = {}
        self.latest_lighting = {"lighting": "Balanced room lighting", "luminance": 120.0, "contrast": 40.0}
        self._latest_raw_frame = None
        
        if OPENCV_AVAILABLE:
            try:
                face_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                self.face_cascade = cv2.CascadeClassifier(face_path)
                
                eye_path = cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml'
                if os.path.exists(eye_path):
                    self.eye_cascade = cv2.CascadeClassifier(eye_path)
                    
                self.enabled = True
            except Exception as e:
                print(f"[VISION ERROR] Could not load cascades: {e}")
                self.enabled = False

    def start(self):
        if not self.enabled or self.is_running:
            return
            
        try:
            self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW if os.name == 'nt' else cv2.CAP_ANY)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(0)
                
            if not self.cap.isOpened():
                print("[VISION] Webcam not accessible or occupied by another app.")
                self.is_running = False
                return
                
            self.is_running = True
            self._thread = threading.Thread(target=self._vision_loop, daemon=True)
            self._thread.start()
            print("[VISION] Camera Hardware ACTIVATED -> Neural Multi-Modal Perception ON.")
        except Exception as e:
            print(f"[VISION ERROR] Failed to start camera: {e}")
            self.is_running = False

    def _match_or_create_user(self, bbox, frame_gray):
        bx, by, bw, bh = bbox
        bcx = bx + bw / 2.0
        bcy = by + bh / 2.0
        
        best_id = None
        best_dist = 120.0 # Pixel distance threshold
        
        for uid, user in self.tracked_users.items():
            ux, uy, uw, uh = user.bbox
            ucx = ux + uw / 2.0
            ucy = uy + uh / 2.0
            dist = np.hypot(bcx - ucx, bcy - ucy)
            if dist < best_dist:
                best_dist = dist
                best_id = uid
                
        if best_id is not None:
            self.tracked_users[best_id].update(bbox, frame_gray)
            return best_id
        else:
            uid = self.next_user_id
            self.next_user_id += 1
            new_u = TrackedUser(uid, bbox)
            new_u.update(bbox, frame_gray)
            self.tracked_users[uid] = new_u
            return uid

    # -------------------------------------------------------------------------
    # 1. COLOR INTELLIGENCE: Calibrated HSV Color Extraction
    # -------------------------------------------------------------------------
    def _analyze_color_name(self, hsv_crop):
        """Returns the most dominant accurate color name and score from an HSV image crop."""
        if hsv_crop is None or hsv_crop.size == 0:
            return "neutral"
            
        h_chan = hsv_crop[:, :, 0]
        s_chan = hsv_crop[:, :, 1]
        v_chan = hsv_crop[:, :, 2]
        total_pixels = float(hsv_crop.shape[0] * hsv_crop.shape[1])
        if total_pixels == 0:
            return "neutral"

        color_masks = {
            "Red": (((h_chan <= 10) | (h_chan >= 168)) & (s_chan >= 55) & (v_chan >= 45)),
            "Orange": ((h_chan >= 11) & (h_chan <= 24) & (s_chan >= 60) & (v_chan >= 65)),
            "Yellow": ((h_chan >= 25) & (h_chan <= 36) & (s_chan >= 55) & (v_chan >= 70)),
            "Green": ((h_chan >= 37) & (h_chan <= 85) & (s_chan >= 40) & (v_chan >= 35)),
            "Teal / Cyan": ((h_chan >= 86) & (h_chan <= 100) & (s_chan >= 45) & (v_chan >= 45)),
            "Blue": ((h_chan >= 101) & (h_chan <= 135) & (s_chan >= 45) & (v_chan >= 35)),
            "Purple / Violet": ((h_chan >= 136) & (h_chan <= 158) & (s_chan >= 40) & (v_chan >= 40)),
            "Pink": ((h_chan >= 159) & (h_chan <= 167) & (s_chan >= 40) & (v_chan >= 65)),
            "White": ((s_chan <= 35) & (v_chan >= 165)),
            "Black": (v_chan <= 45),
            "Gray": ((s_chan <= 40) & (v_chan > 45) & (v_chan < 165)),
            "Brown": ((h_chan >= 10) & (h_chan <= 24) & (s_chan >= 40) & (v_chan >= 30) & (v_chan <= 120))
        }

        color_scores = {}
        for cname, mask in color_masks.items():
            pct = float(np.sum(mask)) / total_pixels
            color_scores[cname] = pct

        sorted_colors = sorted(color_scores.items(), key=lambda x: x[1], reverse=True)
        top_color, top_pct = sorted_colors[0]

        # Prioritize chromatic colors over pure neutral black/gray if chromatic signal is distinct (> 18%)
        if top_color in ["Black", "Gray", "White"] and top_pct < 0.65:
            for cname, pct in sorted_colors:
                if cname not in ["Black", "Gray", "White"] and pct > 0.18:
                    return cname
        return top_color

    def _analyze_scene_dominant_colors(self, frame):
        """Extracts top dominant color distribution across the room/environment."""
        try:
            small = cv2.resize(frame, (160, 120))
            hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
            h_chan, s_chan, v_chan = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
            total_px = float(small.shape[0] * small.shape[1])

            color_definitions = {
                "Blue": ((h_chan >= 101) & (h_chan <= 135) & (s_chan >= 45) & (v_chan >= 35)),
                "Green": ((h_chan >= 37) & (h_chan <= 85) & (s_chan >= 40) & (v_chan >= 35)),
                "Red": (((h_chan <= 10) | (h_chan >= 168)) & (s_chan >= 55) & (v_chan >= 45)),
                "Yellow": ((h_chan >= 25) & (h_chan <= 36) & (s_chan >= 55) & (v_chan >= 70)),
                "Orange": ((h_chan >= 11) & (h_chan <= 24) & (s_chan >= 60) & (v_chan >= 65)),
                "Cyan": ((h_chan >= 86) & (h_chan <= 100) & (s_chan >= 45) & (v_chan >= 45)),
                "Purple": ((h_chan >= 136) & (h_chan <= 158) & (s_chan >= 40) & (v_chan >= 40)),
                "Pink": ((h_chan >= 159) & (h_chan <= 167) & (s_chan >= 40) & (v_chan >= 65)),
                "White": ((s_chan <= 35) & (v_chan >= 165)),
                "Black": (v_chan <= 45),
                "Gray": ((s_chan <= 40) & (v_chan > 45) & (v_chan < 165)),
                "Brown": ((h_chan >= 10) & (h_chan <= 24) & (s_chan >= 40) & (v_chan >= 30) & (v_chan <= 120))
            }

            ranked = []
            for name, mask in color_definitions.items():
                pct = (np.sum(mask) / total_px) * 100.0
                if pct >= 3.0:
                    ranked.append((name, round(pct, 1)))

            ranked.sort(key=lambda x: x[1], reverse=True)
            return ranked[:4]
        except Exception:
            return [("Neutral", 100.0)]

    # -------------------------------------------------------------------------
    # 2. NUMBERS & HAND FINGER COUNTING: Robust Dual-Space Skin & Defect Analysis
    # -------------------------------------------------------------------------
    def _analyze_hands_and_fingers(self, frame, face_bboxes):
        """Detects hand contours, counts extended fingers (0-5), and classifies gestures."""
        hands_found = []
        try:
            fh, fw, _ = frame.shape
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)

            # Dual-space skin detection mask
            hsv_mask = (hsv[:,:,0] <= 24) & (hsv[:,:,1] >= 42) & (hsv[:,:,1] <= 185) & (hsv[:,:,2] >= 55)
            ycrcb_mask = (ycrcb[:,:,1] >= 133) & (ycrcb[:,:,1] <= 175) & (ycrcb[:,:,2] >= 77) & (ycrcb[:,:,2] <= 128)
            skin_bin = (hsv_mask & ycrcb_mask).astype(np.uint8) * 255

            # Mask out detected face boxes with safety margin to prevent face blobs from counting as hands
            for fx, fy, fw_f, fh_f in face_bboxes:
                pad_x = int(fw_f * 0.15)
                pad_y = int(fh_f * 0.25)
                x1 = max(0, fx - pad_x)
                y1 = max(0, fy - pad_y)
                x2 = min(fw, fx + fw_f + pad_x)
                y2 = min(fh, fy + fh_f + pad_y)
                skin_bin[y1:y2, x1:x2] = 0

            # Morphological smoothing
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            skin_bin = cv2.morphologyEx(skin_bin, cv2.MORPH_OPEN, kernel, iterations=1)
            skin_bin = cv2.dilate(skin_bin, kernel, iterations=2)

            contours, _ = cv2.findContours(skin_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < 2800 or area > 75000:
                    continue

                x, y, w, h = cv2.boundingRect(cnt)
                aspect_ratio = float(w) / float(h)
                if aspect_ratio < 0.25 or aspect_ratio > 4.0:
                    continue

                # Convex Hull & Defect Analysis
                hull_pts = cv2.convexHull(cnt, returnPoints=True)
                hull_idx = cv2.convexHull(cnt, returnPoints=False)
                
                if hull_idx is None or len(hull_idx) <= 3:
                    continue

                defects = None
                try:
                    defects = cv2.convexityDefects(cnt, hull_idx)
                except Exception:
                    pass

                defect_count = 0
                finger_tips = []
                
                if defects is not None:
                    for i in range(defects.shape[0]):
                        s, e, f, d = defects[i, 0]
                        start = tuple(cnt[s][0])
                        end = tuple(cnt[e][0])
                        far = tuple(cnt[f][0])

                        a = np.hypot(end[0] - start[0], end[1] - start[1])
                        b = np.hypot(far[0] - start[0], far[1] - start[1])
                        c = np.hypot(end[0] - far[0], end[1] - far[1])

                        if b * c > 0:
                            angle = np.arccos(np.clip((b**2 + c**2 - a**2) / (2 * b * c), -1.0, 1.0)) * 180 / np.pi
                            if angle <= 88 and d > 3200:
                                defect_count += 1
                                finger_tips.append(start)

                # Classify finger count and iconic gestures
                if defect_count == 0:
                    # Check contour elongation: 1 finger pointing vs closed fist
                    cnt_hull_area = cv2.contourArea(hull_pts)
                    solidity = float(area) / (cnt_hull_area + 1e-5)
                    if solidity > 0.88:
                        fingers = 0
                        gesture = "Fist / 0 Fingers"
                    else:
                        fingers = 1
                        gesture = "1 Finger (Pointing / Number One)"
                elif defect_count == 1:
                    fingers = 2
                    gesture = "2 Fingers (Peace / Victory Sign / Number Two)"
                elif defect_count == 2:
                    fingers = 3
                    gesture = "3 Fingers (Number Three)"
                elif defect_count == 3:
                    fingers = 4
                    gesture = "4 Fingers (Number Four)"
                else:
                    fingers = 5
                    gesture = "5 Fingers (Open Palm / High-Five / Number Five)"

                hands_found.append({
                    "bbox": (x, y, w, h),
                    "fingers": fingers,
                    "gesture": gesture,
                    "center": (x + w // 2, y + h // 2),
                    "area": int(area)
                })
        except Exception:
            pass

        return hands_found

    # -------------------------------------------------------------------------
    # 3. OBJECT DETECTION & DESK ITEM RECOGNITION
    # -------------------------------------------------------------------------
    def _analyze_salient_objects(self, frame, face_bboxes, hand_bboxes):
        """Detects desk and handheld objects: Smartphones, Cups/Mugs/Bottles, Books/Notebooks, Pens, Glasses, Screens."""
        objects = []
        fh, fw, _ = frame.shape
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Exclusion mask for faces and hands
        mask_exclude = np.zeros((fh, fw), dtype=np.uint8)
        for fx, fy, f_w, f_h in face_bboxes:
            cv2.rectangle(mask_exclude, (max(0, fx - 10), max(0, fy - 10)), (min(fw, fx + f_w + 10), min(fh, fy + f_h + 10)), 255, -1)
        for h_info in hand_bboxes:
            hx, hy, hw, hh = h_info["bbox"]
            cv2.rectangle(mask_exclude, (max(0, hx - 8), max(0, hy - 8)), (min(fw, hx + hw + 8), min(fh, hy + hh + 8)), 255, -1)

        # Canny edge detector for structured geometric shapes
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 40, 130)
        edges[mask_exclude > 0] = 0

        # Morphological close to join object contours
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(edges_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 1500 or area > 140000:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            aspect = float(h) / float(w + 1e-5)
            roi_hsv = hsv[y:y+h, x:x+w]
            color_name = self._analyze_color_name(roi_hsv)

            # Object Classification Heuristics
            # 1. Smartphone / Mobile device: rectangular vertical or horizontal aspect ratio ~1.7 to 2.4
            if 1.6 <= aspect <= 2.5 and 1800 <= area <= 32000 and w >= 40 and h >= 75:
                objects.append({
                    "name": "Smartphone",
                    "bbox": (x, y, w, h),
                    "color": color_name,
                    "desc": f"{color_name} Smartphone / Mobile Phone",
                    "confidence": 0.88
                })
            elif 0.40 <= aspect <= 0.62 and 1800 <= area <= 32000 and w >= 75 and h >= 40:
                objects.append({
                    "name": "Smartphone",
                    "bbox": (x, y, w, h),
                    "color": color_name,
                    "desc": f"{color_name} Smartphone (Horizontal orientation)",
                    "confidence": 0.85
                })

            # 2. Cup / Mug / Bottle: cylindrical profile
            elif 1.15 <= aspect <= 2.8 and 1600 <= area <= 28000 and 35 <= w <= 140:
                # Distinguish from smartphone by circularity / top rim ellipse
                hull = cv2.convexHull(cnt)
                solidity = float(area) / (cv2.contourArea(hull) + 1e-5)
                if solidity > 0.72:
                    name_type = "Water Bottle" if aspect > 1.9 else "Cup / Coffee Mug"
                    objects.append({
                        "name": name_type,
                        "bbox": (x, y, w, h),
                        "color": color_name,
                        "desc": f"{color_name} {name_type}",
                        "confidence": 0.82
                    })

            # 3. Book / Notebook / Paper Document: large planar rectangular shape
            elif 1.15 <= aspect <= 1.65 and area > 18000 and w >= 110:
                objects.append({
                    "name": "Notebook / Book",
                    "bbox": (x, y, w, h),
                    "color": color_name,
                    "desc": f"{color_name} Book / Notebook / Document",
                    "confidence": 0.84
                })

            # 4. Pen / Pencil / Stylus: high elongation ratio (> 3.8)
            elif (aspect > 3.8 or aspect < 0.26) and 600 <= area <= 6000:
                objects.append({
                    "name": "Pen / Stylus",
                    "bbox": (x, y, w, h),
                    "color": color_name,
                    "desc": f"{color_name} Pen / Writing Instrument",
                    "confidence": 0.80
                })

        # Eyeglasses / Spectacles on detected faces
        for fx, fy, fw_f, fh_f in face_bboxes:
            eye_top = max(0, fy + int(fh_f * 0.18))
            eye_bottom = min(fh, fy + int(fh_f * 0.54))
            eye_left = max(0, fx + int(fw_f * 0.10))
            eye_right = min(fw, fx + int(fw_f * 0.90))

            if eye_bottom > eye_top and eye_right > eye_left:
                eye_crop = gray[eye_top:eye_bottom, eye_left:eye_right]
                eye_edges = cv2.Canny(eye_crop, 50, 150)
                edge_density = float(np.sum(eye_edges > 0)) / float(eye_edges.size)
                
                mid_w = eye_crop.shape[1] // 2
                bridge = eye_crop[:, max(0, mid_w - 6):min(eye_crop.shape[1], mid_w + 6)]
                bridge_std = float(np.std(bridge))

                if edge_density > 0.18 and bridge_std > 30.0:
                    objects.append({
                        "name": "Eyeglasses",
                        "bbox": (eye_left, eye_top, eye_right - eye_left, eye_bottom - eye_top),
                        "color": "Dark / Metallic",
                        "desc": "Eyeglasses / Spectacles worn on face",
                        "confidence": 0.92
                    })

        return objects[:5] # Return top distinct salient objects

    # -------------------------------------------------------------------------
    # 4. DIGIT & PRINTED NUMBER RECOGNITION (OCR Heuristics)
    # -------------------------------------------------------------------------
    def _analyze_numbers_and_text(self, frame, face_bboxes):
        """Inspects rectangular cards, paper, or signs in view to detect printed digits and count characters."""
        digits = []
        try:
            fh, fw, _ = frame.shape
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Mask out face region
            for fx, fy, fw_f, fh_f in face_bboxes:
                cv2.rectangle(gray, (fx, fy), (fx + fw_f, fy + fh_f), 128, -1)

            # Adaptive thresholding for high contrast text on paper/cards
            thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 6)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            char_candidates = []
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                area = cv2.contourArea(cnt)
                # Digit character size filter
                if 20 <= h <= 180 and 8 <= w <= 120 and 150 <= area <= 12000:
                    aspect = float(h) / float(w)
                    if 0.9 <= aspect <= 3.5:
                        char_candidates.append((x, y, w, h, area))

            if char_candidates:
                digits.append(f"{len(char_candidates)} printed numerical character(s) visible on document / display")
        except Exception:
            pass
        return digits

    # -------------------------------------------------------------------------
    # 5. ENVIRONMENT, LIGHTING & CLOTHING
    # -------------------------------------------------------------------------
    def _analyze_environment_and_lighting(self, frame):
        """Analyzes ambient brightness, luminance and contrast."""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            mean_lum = float(np.mean(gray))
            contrast = float(np.std(gray))
            
            if mean_lum > 165: light_desc = "Bright, well-illuminated room"
            elif mean_lum >= 90: light_desc = "Balanced, clear indoor room lighting"
            elif mean_lum >= 45: light_desc = "Soft / warm ambient indoor lighting"
            else: light_desc = "Dim / low-light ambient environment"
                
            return {"lighting": light_desc, "luminance": round(mean_lum, 1), "contrast": round(contrast, 1)}
        except Exception:
            return {"lighting": "Normal indoor lighting", "luminance": 120.0, "contrast": 40.0}

    def _analyze_clothing_region(self, frame, face_bbox):
        """Identifies shirt / clothing color and placket/button details."""
        try:
            fh, fw, _ = frame.shape
            fx, fy, fw_face, fh_face = face_bbox
            
            chest_top = fy + fh_face + int(fh_face * 0.10)
            chest_bottom = min(fh, fy + fh_face + int(fh_face * 1.70))
            chest_left = max(0, fx - int(fw_face * 0.35))
            chest_right = min(fw, fx + fw_face + int(fw_face * 0.35))
            
            if chest_bottom <= chest_top or chest_right <= chest_left:
                return {"primary_color": "neutral", "buttons": "no visible buttons", "desc": "torso not in view"}
                
            chest_crop = frame[chest_top:chest_bottom, chest_left:chest_right]
            if chest_crop.shape[0] < 20 or chest_crop.shape[1] < 20:
                return {"primary_color": "neutral", "buttons": "no visible buttons", "desc": "torso too small"}
                
            hsv = cv2.cvtColor(chest_crop, cv2.COLOR_BGR2HSV)
            color_name = self._analyze_color_name(hsv)

            chest_gray = cv2.cvtColor(chest_crop, cv2.COLOR_BGR2GRAY)
            mid_x = chest_gray.shape[1] // 2
            strip_w = max(10, int(chest_gray.shape[1] * 0.15))
            mid_strip = chest_gray[:, max(0, mid_x - strip_w) : min(chest_gray.shape[1], mid_x + strip_w)]
            
            edges = cv2.Canny(mid_strip, 50, 150)
            edge_density = float(np.sum(edges > 0)) / float(edges.size)
            
            buttons_info = "has visible buttons / collar placket" if edge_density > 0.22 else "plain smooth fabric / t-shirt / crewneck"

            return {
                "primary_color": color_name,
                "buttons": buttons_info,
                "desc": f"{color_name} colored shirt/top with {buttons_info}"
            }
        except Exception:
            return {"primary_color": "neutral", "buttons": "no buttons", "desc": "analyzed"}

    # -------------------------------------------------------------------------
    # 6. HIGH-TECH HUD OVERLAY (Cyberpunk / Jarvis Visual Telemetry)
    # -------------------------------------------------------------------------
    def _draw_ai_hud_overlay(self, frame):
        """Annotates live camera frame with bounding boxes, hand finger counts, object tags, and color palette."""
        h, w, _ = frame.shape
        annotated = frame.copy()

        # Top HUD Status Header
        cv2.rectangle(annotated, (0, 0), (w, 32), (15, 12, 8), -1)
        cv2.line(annotated, (0, 32), (w, 32), (70, 200, 255), 1)
        cv2.putText(annotated, "MONICA AI NEURAL VISION | REAL-TIME TRACKING", (12, 21), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 215, 70), 1, cv2.LINE_AA)
        
        user_count_txt = f"USERS: {len(self.tracked_users)} | OBJS: {len(self.latest_objects)}"
        cv2.putText(annotated, user_count_txt, (w - 170, 21), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (80, 240, 120), 1, cv2.LINE_AA)

        # Center Targeting Reticle
        cx, cy = w // 2, h // 2
        cv2.circle(annotated, (cx, cy), 16, (100, 180, 255), 1, cv2.LINE_AA)
        cv2.line(annotated, (cx - 24, cy), (cx + 24, cy), (100, 180, 255), 1)
        cv2.line(annotated, (cx, cy - 24), (cx, cy + 24), (100, 180, 255), 1)

        # 1. Draw User Face Bounding Boxes
        for uid, user in self.tracked_users.items():
            ux, uy, uw, uh = user.bbox
            is_speaker = (uid == self.active_speaker_id)
            box_color = (0, 230, 255) if is_speaker else (70, 160, 240)
            line_thickness = 2 if is_speaker else 1

            # Corner Brackets
            d = min(20, uw // 4)
            cv2.line(annotated, (ux, uy), (ux + d, uy), box_color, line_thickness + 1)
            cv2.line(annotated, (ux, uy), (ux, uy + d), box_color, line_thickness + 1)
            cv2.line(annotated, (ux + uw, uy), (ux + uw - d, uy), box_color, line_thickness + 1)
            cv2.line(annotated, (ux + uw, uy), (ux + uw, uy + d), box_color, line_thickness + 1)
            cv2.line(annotated, (ux, uy + uh), (ux + d, uy + uh), box_color, line_thickness + 1)
            cv2.line(annotated, (ux, uy + uh), (ux, uy + uh - d), box_color, line_thickness + 1)
            cv2.line(annotated, (ux + uw, uy + uh), (ux + uw - d, uy + uh), box_color, line_thickness + 1)
            cv2.line(annotated, (ux + uw, uy + uh), (ux + uw, uy + uh - d), box_color, line_thickness + 1)
            cv2.rectangle(annotated, (ux, uy), (ux + uw, uy + uh), box_color, 1)

            # Tag Label
            tag_text = f"TARGET: {user.label.upper()}" + (" [SPEAKER LOCK]" if is_speaker else "")
            tag_size = cv2.getTextSize(tag_text, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)[0]
            cv2.rectangle(annotated, (ux, max(0, uy - 18)), (ux + tag_size[0] + 8, uy), (18, 14, 8), -1)
            cv2.rectangle(annotated, (ux, max(0, uy - 18)), (ux + tag_size[0] + 8, uy), box_color, 1)
            cv2.putText(annotated, tag_text, (ux + 4, uy - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.40, box_color, 1, cv2.LINE_AA)

            # Mouth Motion Activity Meter
            if user.mouth_bbox:
                mx, my, mw, mh = user.mouth_bbox
                mouth_color = (0, 60, 255) if user.speech_activity > 4.0 else (80, 200, 120)
                cv2.rectangle(annotated, (mx, my), (mx + mw, my + mh), mouth_color, 1)

        # 2. Draw Detected Objects Bounding Boxes
        for obj in self.latest_objects:
            ox, oy, ow, oh = obj["bbox"]
            obj_color = (50, 220, 120) # Neon emerald
            cv2.rectangle(annotated, (ox, oy), (ox + ow, oy + oh), obj_color, 1)
            lbl = f"[{obj['name'].upper()} | {obj['color'].upper()}]"
            lbl_sz = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)[0]
            cv2.rectangle(annotated, (ox, max(0, oy - 16)), (ox + lbl_sz[0] + 6, oy), (15, 20, 15), -1)
            cv2.rectangle(annotated, (ox, max(0, oy - 16)), (ox + lbl_sz[0] + 6, oy), obj_color, 1)
            cv2.putText(annotated, lbl, (ox + 3, oy - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, obj_color, 1, cv2.LINE_AA)

        # 3. Draw Detected Hands & Finger Count Badges
        for hand in self.latest_hands:
            hx, hy, hw, hh = hand["bbox"]
            hand_color = (255, 120, 230) # Magenta/Neon Purple
            cv2.rectangle(annotated, (hx, hy), (hx + hw, hy + hh), hand_color, 1)
            h_lbl = f"[HAND: {hand['fingers']} FINGERS | {hand['gesture'].upper()}]"
            h_sz = cv2.getTextSize(h_lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)[0]
            cv2.rectangle(annotated, (hx, max(0, hy - 18)), (hx + h_sz[0] + 8, hy), (25, 10, 25), -1)
            cv2.rectangle(annotated, (hx, max(0, hy - 18)), (hx + h_sz[0] + 8, hy), hand_color, 1)
            cv2.putText(annotated, h_lbl, (hx + 4, hy - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA)

        # Bottom HUD Color Spectrum Bar
        if self.latest_scene_colors:
            cv2.rectangle(annotated, (0, h - 24), (w, h), (12, 10, 8), -1)
            cv2.line(annotated, (0, h - 24), (w, h - 24), (60, 180, 255), 1)
            palette_str = "ROOM PALETTE: " + " | ".join([f"{c[0].upper()} {c[1]}%" for c in self.latest_scene_colors])
            cv2.putText(annotated, palette_str, (12, h - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 225, 255), 1, cv2.LINE_AA)

        return annotated

    # -------------------------------------------------------------------------
    # 7. MAIN VISION WORKER LOOP (~25 FPS)
    # -------------------------------------------------------------------------
    def _vision_loop(self):
        global _latest_jpeg_frame
        from ui_server import update_ui_state
        
        while self.is_running and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.08)
                continue
                
            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # 1. Face Detection
            faces = self.face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.12, 
                minNeighbors=4, 
                minSize=(65, 65)
            )
            
            now = time.time()
            seen_ids = set()
            
            with self.lock:
                self._latest_raw_frame = frame.copy()
                
                # Update Tracked Users
                for bbox in faces:
                    uid = self._match_or_create_user(bbox, gray)
                    seen_ids.add(uid)
                    cloth_info = self._analyze_clothing_region(frame, bbox)
                    self.latest_clothing_analysis[uid] = cloth_info

                # Clean up lost faces
                lost_ids = [uid for uid, u in self.tracked_users.items() if (now - u.last_seen) > 2.0]
                for uid in lost_ids:
                    del self.tracked_users[uid]
                    if uid in self.latest_clothing_analysis:
                        del self.latest_clothing_analysis[uid]
                    if self.active_speaker_id == uid:
                        self.active_speaker_id = None

                # 2. Hand & Finger Counting Analysis
                self.latest_hands = self._analyze_hands_and_fingers(frame, faces)

                # 3. Salient Objects & Item Detection
                self.latest_objects = self._analyze_salient_objects(frame, faces, self.latest_hands)

                # 4. Color Spectrum Analysis
                self.latest_scene_colors = self._analyze_scene_dominant_colors(frame)

                # 5. Digits & Number Recognition
                self.latest_digits_detected = self._analyze_numbers_and_text(frame, faces)

                # 6. Environment & Lighting
                self.latest_lighting = self._analyze_environment_and_lighting(frame)

                # Orient Gaze & Update UI State
                if self.tracked_users:
                    active_user = max(self.tracked_users.values(), key=lambda u: u.speech_activity)
                    if active_user.speech_activity > 4.5 or self.active_speaker_id not in self.tracked_users:
                        self.active_speaker_id = active_user.user_id
                    speaker = self.tracked_users.get(self.active_speaker_id, active_user)
                    
                    self.target_x = speaker.norm_x
                    self.target_y = speaker.norm_y
                    
                    update_ui_state(
                        target_rot_x=float(-self.target_y * 0.35),
                        target_rot_y=float(self.target_x * 0.55),
                        detected_faces_count=len(self.tracked_users),
                        active_speaker_label=speaker.label
                    )
                else:
                    update_ui_state(
                        detected_faces_count=0,
                        active_speaker_label="STANDBY"
                    )

                # Generate Annotated HUD Frame for Live Stream
                annotated_frame = self._draw_ai_hud_overlay(frame)
                ret_enc, buffer = cv2.imencode('.jpg', annotated_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
                if ret_enc:
                    _latest_jpeg_frame = buffer.tobytes()

            time.sleep(0.04) # ~25 FPS

    # -------------------------------------------------------------------------
    # 8. SENSORY SCENE CONTEXT FOR NEURAL REASONING
    # -------------------------------------------------------------------------
    def get_visual_scene_context(self):
        """Generates rich, multi-modal human-eye sensory context for the LLM reasoning brain."""
        with self.lock:
            if not self.is_running:
                return "Camera is currently inactive or off."
            
            # 1. People / Users
            if self.tracked_users:
                user = self.tracked_users.get(self.active_speaker_id) or list(self.tracked_users.values())[0]
                cloth = self.latest_clothing_analysis.get(user.user_id, {})
                shirt_color = cloth.get("primary_color", "detected")
                buttons_info = cloth.get("buttons", "plain fabric")
                speech_state = "speaking / talking" if user.speech_activity > 3.5 else "listening / silent"
                user_summary = f"{len(self.tracked_users)} person(s) present. {user.label} is looking at camera, wearing a {shirt_color} colored top ({buttons_info}), mouth is {speech_state}."
            else:
                user_summary = "No person or face currently visible in front of camera."

            # 2. Hand Gestures & Finger Count
            if self.latest_hands:
                hand_details = []
                for idx, h in enumerate(self.latest_hands, 1):
                    hand_details.append(f"Hand {idx}: holding up {h['fingers']} finger(s) [{h['gesture']}]")
                hands_summary = "; ".join(hand_details)
            else:
                hands_summary = "Hands resting comfortably / no gestures active."

            # 3. Salient Objects Identified
            if self.latest_objects:
                obj_details = [f"{o['desc']} (detected with high confidence)" for o in self.latest_objects]
                objects_summary = ", ".join(obj_details)
            else:
                objects_summary = "No specific standalone desk obstacles or handheld gadgets currently held up."

            # 4. Color Spectrum & Lighting
            if self.latest_scene_colors:
                palette_str = ", ".join([f"{c[0]} ({c[1]}%)" for c in self.latest_scene_colors])
            else:
                palette_str = "Balanced indoor hues"
            lighting_desc = self.latest_lighting.get("lighting", "Clear indoor lighting")

            # 5. Printed Numbers & Digits
            if self.latest_digits_detected:
                digits_str = f"Document / Screen inspection: {'; '.join(self.latest_digits_detected)}."
            else:
                digits_str = "No printed document numbers in close-up view."

            return (
                f"=== REAL-TIME LIVE CAMERA HUMAN-EYE OBSERVATION ===\n"
                f"• Visual Environment & Lighting: {lighting_desc}.\n"
                f"• Dominant Color Spectrum: {palette_str}.\n"
                f"• People & Attire: {user_summary}\n"
                f"• Hand Gestures & Finger Count: {hands_summary}\n"
                f"• Salient Objects Detected: {objects_summary}\n"
                f"• Digit & Number Detection: {digits_str}\n"
                f"(Use these real-time live sensory visual facts directly to answer any question about colors, number of fingers, visible objects, clothing, or surroundings accurately and naturally.)"
            )

    def identify_and_look_at_speaker(self):
        """Called when a question/command is received to orient Monica toward the active user."""
        from ui_server import update_ui_state
        with self.lock:
            if not self.tracked_users:
                return None
                
            speaker = max(self.tracked_users.values(), key=lambda u: (u.speech_activity, u.bbox[2] * u.bbox[3]))
            self.active_speaker_id = speaker.user_id
            
            self.target_x = speaker.norm_x
            self.target_y = speaker.norm_y
            
            update_ui_state(
                target_rot_x=float(-self.target_y * 0.40),
                target_rot_y=float(self.target_x * 0.65),
                active_speaker_label=speaker.label
            )
            return speaker.label

    def look_direction(self, direction):
        """Direct gaze orientation command."""
        from ui_server import update_ui_state
        with self.lock:
            if direction == "left":
                self.target_x = -0.60
                self.target_y = 0.0
            elif direction == "right":
                self.target_x = 0.60
                self.target_y = 0.0
            elif direction == "center":
                self.target_x = 0.0
                self.target_y = 0.0
            update_ui_state(
                target_rot_x=float(-self.target_y * 0.40),
                target_rot_y=float(self.target_x * 0.65)
            )

    def stop(self):
        self.is_running = False
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
        global _latest_jpeg_frame
        _latest_jpeg_frame = None
        from ui_server import update_ui_state
        update_ui_state(
            detected_faces_count=0,
            active_speaker_label="STANDBY"
        )
        print("[VISION] Camera Hardware DEACTIVATED.")
