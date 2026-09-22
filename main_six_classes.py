# ============================================================

# Facial Emotion Recognition Model
# TinyVGG-like CNN trained on FER2013

"""
Real-Time Facial Emotion Recognition

This script performs real-time facial emotion recognition using:
- Haar Cascade for face detection
- MOSSE tracker for face tracking
- A TinyVGG-like CNN trained on FER2013
- Softmax-based confidence estimation

The system recognizes six emotion classes:
angry, fear, happy, neutral, sad, surprise.
"""

# ============================================================

import cv2
import time
import torch
from torch import nn
from torchvision import transforms
from collections import deque
import numpy as np

# ==========================================
# Device
device = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================================
# CNN (TinyVGG)
class EmotionCNN(nn.Module):
    def __init__(self, input_shape, hidden_units, output_shape):
        super().__init__()

        self.block_1 = nn.Sequential(
            nn.Conv2d(input_shape, hidden_units, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.block_2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_units * 12 * 12, output_shape)
        )

    def forward(self, x):
        x = self.block_1(x)
        x = self.block_2(x)
        return self.classifier(x)

# ==========================================
def score_color(value, low, high):

    if value >= high:
        return (0,255,0)

    elif value >= low:
        return (0,255,255)

    else:
        return (0,0,255)

# ==========================================
# Classes
class_names = ['angry','fear','happy','neutral','sad','surprise']

# ==========================================
# Load model
model = EmotionCNN(1,128,len(class_names)).to(device)

MODEL_PATH = "path/to/fer_weights_v5.pth"

model.load_state_dict(
    torch.load(MODEL_PATH, map_location=device)
)

model.eval()

# ==========================================
# Preprocessing
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((48,48)),
    transforms.Grayscale(1),
    transforms.ToTensor(),
    transforms.Normalize([0.5],[0.5])
])

# ==========================================
# Haar Cascade
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ==========================================
# Webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Unable to open webcam.")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# ==========================================
# Tracker
tracker = None
tracking = False

# ==========================================
# Optimisation params
DETECT_EVERY_N_FRAMES = 10
PREDICT_EVERY_N_FRAMES = 3
DETECTION_SCALE = 0.5

# ==========================================
# Thresholds
CONF_LOW, CONF_HIGH = 0.5,0.75
STAB_LOW, STAB_HIGH = 0.4,0.7

# ==========================================
# Metrics

prev_time=time.time()

fps_list=[]
latency_list=[]

# FPS caméra réel
cam_frame_count = 0
cam_start_time = time.time()
cam_fps = 0
cam_fps_list=[]

prediction_history = deque(maxlen = 15)

last_emotion="..."
confidence = 0
stability = 0

frame_count = 0

print("Appuyer sur Q pour quitter")

# ==========================================
# Utils
def valid_box(x, y, w, h, shape):

    H, W = shape

    return w > 30 and h > 30 and x >= 0 and y >= 0 and x + w < W and y + h < H

# ==========================================
# Main loop

while True:

    capture_time = time.time()

    ret,frame = cap.read()

    if not ret:
        break

    frame_count+=1

    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    gray=cv2.equalizeHist(gray)

    # ======================================
    # DETECTION

    if not tracking and frame_count % DETECT_EVERY_N_FRAMES == 0:

        small_gray = cv2.resize(gray, None, fx=DETECTION_SCALE, fy = DETECTION_SCALE)

        faces = face_cascade.detectMultiScale(
            small_gray,
            scaleFactor = 1.3,
            minNeighbors = 6,
            minSize=(30,30)
        )

        if len(faces)  >0:

            x, y, w, h = faces[0]

            x, y, w, h = [int(v/DETECTION_SCALE) for v in (x, y, w, h)]

            tracker  =cv2.legacy.TrackerMOSSE_create()

            tracker.init(frame, (x, y ,w ,h))

            tracking = True

    # ======================================
    # TRACKING

    if tracking:

        success, box=tracker.update(frame)

        if not success or not valid_box(*map(int, box), gray.shape):

            tracking = False

        else:

            x, y ,w ,h = map(int, box)

            size = max(w,h)

            cx, cy = x+w//2,y+h//2

            x1 = max(cx-size//2,0)
            y1 = max(cy-size//2,0)

            x2 = min(cx+size//2,gray.shape[1])
            y2 = min(cy+size//2,gray.shape[0])

            face_roi  =gray[y1:y2, x1:x2]

            if face_roi.size>0 and frame_count % PREDICT_EVERY_N_FRAMES == 0:

                face_tensor = transform(face_roi).unsqueeze(0).to(device)

                with torch.inference_mode():

                    logits = model(face_tensor)

                    probs = torch.softmax(logits, dim=1)

                top2 = torch.topk(probs,2)

                pred_idx = top2.indices[0][0].item()

                last_emotion = class_names[pred_idx]

                confidence = top2.values[0][0].item()

            prediction_history.append(last_emotion)

            stability = prediction_history.count(last_emotion)/len(prediction_history)

            cv2.rectangle(frame, (x1, y1),(x2, y2),(255, 255, 255), 2)

            cv2.putText(frame, last_emotion, (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    # ======================================
    # Metrics

    curr_time = time.time()

    frame_time = curr_time-prev_time
    prev_time = curr_time

    # FPS pipeline
    loop_fps = 1/frame_time if frame_time > 0 else 0
    fps_list.append(loop_fps)

    latency =(curr_time-capture_time)*1000
    latency_list.append(latency)

    # FPS caméra réel
    cam_frame_count+=1
    cam_elapsed = curr_time-cam_start_time

    if cam_elapsed >= 1:

        cam_fps = cam_frame_count/cam_elapsed

        cam_fps_list.append(cam_fps)

        cam_frame_count = 0
        cam_start_time = curr_time

    # ======================================
    # Display metrics

    cv2.putText(frame,f"Camera FPS: {cam_fps:.1f}",(10,25),
                cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,0,0),2)

    cv2.putText(frame,f"Latency: {latency:.1f} ms",(10,50),
                cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),2)

    y=80

    cv2.putText(frame,f"Confidence: {confidence:.2f}",(10,y),
            cv2.FONT_HERSHEY_SIMPLEX,0.5,
            score_color(confidence,CONF_LOW,CONF_HIGH),2)

    y+=20

    cv2.putText(frame,f"Stability: {stability:.2f}",(10,y),
            cv2.FONT_HERSHEY_SIMPLEX,0.5,
            score_color(stability,STAB_LOW,STAB_HIGH),2)

    cv2.imshow("Emotion Recognition - Softmax Metrics",frame)

    if cv2.waitKey(1)&0xFF==ord("q"):
        break

# ==========================================
# Final stats

print("\n===== PERFORMANCE =====")

print(f"FPS moyen pipeline : {np.mean(fps_list):.2f}")
print(f"FPS moyen camera   : {np.mean(cam_fps_list):.2f}")

print(f"Latence moyenne    : {np.mean(latency_list):.2f} ms")

cap.release()

cv2.destroyAllWindows()
