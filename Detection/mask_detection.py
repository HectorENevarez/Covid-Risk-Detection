from tensorflow import keras
import numpy as np
import cv2
import os

def mask_detect(mask_points, net, frame):
    wearing_mask = "unkown"
    
    (x, y, w, h) = mask_points
    roi_face = frame[int(y - 2 * w / 5):int(y + 3 * w / 5), x:(x + w)]

    HEIGHT, WIDTH = roi_face.shape[:2]
    if HEIGHT >= 30 and WIDTH >= 30:
        save_loc = os.path.join("./mask_images", '_head_shot.jpg')
        cv2.imwrite(save_loc, roi_face)
        
        mask_shot = keras.preprocessing.image.load_img(save_loc, target_size=(300, 300, 3))
        temp = keras.preprocessing.image.img_to_array(mask_shot)
        temp = temp * 1./255
        temp = np.expand_dims(temp, axis=0)
        prediction = np.argmax(net.predict(temp))
        
        if prediction == 0:
            wearing_mask = "nomask"
        elif prediction == 1:
            wearing_mask = "mask"
        else:
            wearing_mask = "unkown"

    return wearing_mask