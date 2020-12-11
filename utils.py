
import numpy as np
from imutils import face_utils
from tensorflow.keras.models import load_model


class Blink_detection:

    def __init__(self):
        self.IMG_SIZE = (34, 26)
        self.model = load_model('FILES\\blinkdetection.h5') 

    def crop_eye(self, gray, eye_points):
        x1, y1 = np.amin(eye_points, axis=0)
        x2, y2 = np.amax(eye_points, axis=0)
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

        w = (x2 - x1) * 1.2
        h = w * self.IMG_SIZE[1] / self.IMG_SIZE[0]

        margin_x, margin_y = w / 2, h / 2

        min_x, min_y = int(cx - margin_x), int(cy - margin_y)
        max_x, max_y = int(cx + margin_x), int(cy + margin_y)

        eye_rect = np.rint([min_x, min_y, max_x, max_y]).astype(np.int)

        eye_img = gray[eye_rect[1]:eye_rect[3], eye_rect[0]:eye_rect[2]]

        return eye_img, eye_rect  

    def model_predict(self, eye_input_l, eye_input_r ):
        pred_l = self.model.predict(eye_input_l)
        pred_r = self.model.predict(eye_input_r)

        return pred_l, pred_r
         