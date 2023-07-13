import os
import cv2
import numpy as np
from kivy.app import App
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture

WEIGHTS_PATH= "yolo_fastest/yolo-fastest-1.1.weights"
CFG_PATH= "yolo_fastest/yolo-fastest-1.1.cfg"
COCO_NAMES= "yolo_fastest/coco.names"

class MainApp(App):
    def build(self):
        # Load the model
        self.net = cv2.dnn.readNet(WEIGHTS_PATH, CFG_PATH)
        
        self.labelsPath = COCO_NAMES
        self.LABELS = open(self.labelsPath).read().strip().split("\n")
        
        self.img1 = Image()
        self.capture = cv2.VideoCapture('rtsp://admin:admin@192.168.1.7:1935')
        
        Clock.schedule_interval(self.update, 1.0/33.0)

        return self.img1

    def update(self, dt):
        ret, frame = self.capture.read()
        
        # Get the detections
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)
        layerOutputs = self.net.forward(self.net.getUnconnectedOutLayersNames())

        for output in layerOutputs:
            for detection in output:
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]
                
                if confidence > 0.5:
                    box = detection[0:4] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
                    (centerX, centerY, width, height) = box.astype("int")
                    
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
                    cv2.putText(frame, str(self.LABELS[classID]), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
        
        buf1 = cv2.flip(frame, 0)
        buf = buf1.tostring()

        self.img1.texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        self.img1.texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')

    def on_stop(self):
        # close video on app stop (x pressed)
        self.capture.release()

if __name__ == '__main__':
    MainApp().run()
