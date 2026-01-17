import cv2
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
import time
import os
from collections import deque
import argparse

device = torch.device('cpu')

class ImprovedEmotionCNN(nn.Module):
    """model architecture matching the production model"""
    def __init__(self, num_classes=7):
        super(ImprovedEmotionCNN, self).__init__()
        
        # use resnet18 as backbone
        self.backbone = models.resnet18(weights=None)
        self.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.backbone.maxpool = nn.Identity()  # remove maxpool for small images
        
        # replace final layer to get 256 features
        self.backbone.fc = nn.Linear(512, 256)
        
        # additional layers
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(256, num_classes)
    
    def forward(self, x):
        features = self.backbone(x)
        features = self.dropout(features)
        logits = self.classifier(features)
        return logits

class EmotionDetector:
    def __init__(self, model_path=None):
        # emotion labels
        self.emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
        
        # color mapping for emotions
        self.emotion_colors = {
            'Angry': (0, 0, 255),      # red
            'Disgust': (0, 128, 0),    # dark green
            'Fear': (128, 0, 128),     # purple
            'Happy': (0, 255, 255),    # yellow
            'Neutral': (128, 128, 128), # gray
            'Sad': (255, 0, 0),        # blue
            'Surprise': (0, 165, 255)  # orange
        }
        
        # initialize model
        self.model = ImprovedEmotionCNN(num_classes=7)
        self.model.to(device)
        
        if model_path is None:
            model_path = 'emotion_model.pth'
        
        # make sure model path is relative to script directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        if not os.path.isabs(model_path):
            model_path = os.path.join(script_dir, model_path)
        
        if model_path and os.path.exists(model_path):
            print(f"loading model: {model_path}")
            try:
                # use map_location to handle cpu differences
                checkpoint = torch.load(model_path, map_location=device, weights_only=False)
                if 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                    print("model info:")
                    print(f"  original model: {checkpoint.get('original_model', 'unknown')}")
                    print(f"  modification: {checkpoint.get('modification', 'unknown')}")
                    reduction_pct = checkpoint.get('reduction_percentage', 0)
                    if reduction_pct > 0:
                        print(f"  sad reduction: {reduction_pct}%")
                    else:
                        print("  base model (no reduction)")
                else:
                    # direct state dict
                    self.model.load_state_dict(checkpoint)
                    print("model loaded successfully (direct state dict)")
            except Exception as e:
                print(f"error loading model: {e}")
                print("using untrained model")
        else:
            print(f"model file not found: {model_path}")
            print("using untrained model")
        
        self.model.eval()
        
        # image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((48, 48)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # face detection
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # performance tracking
        self.fps_history = deque(maxlen=10)
        self.target_fps = 10
        self.frame_count = 0
        self.emotion_confidence_history = {emotion: deque(maxlen=30) for emotion in self.emotions}
        
        # current emotion state (persists between frames)
        self.current_emotion_idx = None
        self.current_confidence = 0.0
        self.current_probabilities = np.ones(7) / 7
        self.current_face_coords = None
    
    def detect_faces(self, frame):
        """detect faces in frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if self.face_cascade is not None:
            # conservative detection to reduce false positives
            faces = self.face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.2,      # less sensitive scaling
                minNeighbors=6,       # higher threshold for reliability
                minSize=(40, 40),     # larger minimun size
                maxSize=(300, 300),   # maximum size limit
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            # fallback with relaxed parameters
            if len(faces) == 0:
                faces = self.face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.15,  # moderate sensitivity
                    minNeighbors=4,    # still conservative
                    minSize=(30, 30),  # slightly smaller
                    maxSize=(350, 350),
                    flags=cv2.CASCADE_SCALE_IMAGE
                )
        else:
            faces = []
        
        return faces
    
    def predict_emotion(self, face_region):
        """predict emotion from face region with sad bias reduction"""
        try:
            # convert BGR to RGB
            face_rgb = cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB)
            
            # convert to PIL image and apply transforms
            face_pil = Image.fromarray(face_rgb)
            
            # apply transforms
            face_tensor = self.transform(face_pil).unsqueeze(0).to(device)
            
            with torch.no_grad():
                outputs = self.model(face_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                
                # sad bias reduction (60% reduction)
                sad_emotion_idx = 5  # sad is index 5
                neutral_emotion_idx = 4  # nuetral is index 4
                
                # reduce sad probability by 60% and boost neutral
                sad_reduction_factor = 0.4  # reduce sad to 40% of original
                original_sad_prob = probabilities[0][sad_emotion_idx].clone()
                probabilities[0][sad_emotion_idx] *= sad_reduction_factor
                
                # boost neutral by portion of reduced sad
                sad_reduction_amount = original_sad_prob * (1 - sad_reduction_factor)
                probabilities[0][neutral_emotion_idx] += sad_reduction_amount * 0.7
                
                # re-normalize probabilities
                probabilities = probabilities / probabilities.sum(dim=1, keepdim=True)
                
                # get prediction
                predicted_idx = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][predicted_idx].item()
                
                return predicted_idx, confidence, probabilities[0].cpu().numpy()
                
        except Exception as e:
            print(f"error in emotion prediction: {e}")
            return 4, 0.0, np.ones(7) / 7  # return neutral as default
    
    def draw_emotion_info(self, frame, emotion_idx, confidence, probabilities, face_coords):
        """draw emotion information on frame"""
        # handle case where emotion_idx might be None
        if emotion_idx is None:
            emotion_idx = 4  # default to neutral
            confidence = 0.0
            probabilities = np.ones(7) / 7
        
        x, y, w, h = face_coords
        emotion_name = self.emotions[emotion_idx]
        color = self.emotion_colors[emotion_name]
        
        # draw face rectangle with borders
        cv2.rectangle(frame, (x-2, y-2), (x + w + 2, y + h + 2), (255, 255, 255), 1)  # white outer border
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)  # main colored border
        
        # draw emotion label with confidence
        label = f"{emotion_name}: {confidence:.1%}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.8, 2)[0]
        
        # create background for label
        bg_x1, bg_y1 = x - 5, y - label_size[1] - 15
        bg_x2, bg_y2 = x + label_size[0] + 10, y - 2
        cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)  # black background
        cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2), color, 2)  # colored border
        
        # draw label text with shadow
        cv2.putText(frame, label, (x+1, y-7), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 0), 3)  # shadow
        cv2.putText(frame, label, (x, y-8), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)  # main text
        
        # draw emotion probabilities panel
        self._draw_emotion_panel(frame, probabilities)
        
        # update confidence history
        self.emotion_confidence_history[emotion_name].append(confidence)
    
    def _draw_emotion_panel(self, frame, probabilities):
        """draw emotion probability panel"""
        panel_x, panel_y = 20, 50
        panel_width, panel_height = 350, 280
        
        # panel background with transparency
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x-10, panel_y-10), 
                     (panel_x + panel_width + 10, panel_y + panel_height + 10), 
                     (40, 40, 40), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        
        # panel border
        cv2.rectangle(frame, (panel_x-10, panel_y-10), 
                     (panel_x + panel_width + 10, panel_y + panel_height + 10), 
                     (100, 150, 255), 2)
        
        # title
        title = "EMOTION ANALYSIS"
        cv2.putText(frame, title, (panel_x, panel_y-20), cv2.FONT_HERSHEY_DUPLEX, 0.7, (100, 150, 255), 2)
        
        # emotion bars
        bar_width = 250
        bar_height = 20
        
        for i, (emotion, prob) in enumerate(zip(self.emotions, probabilities)):
            y_pos = panel_y + i * 35
            bar_length = int(bar_width * prob)
            
            # emotion name only 
            emotion_text = emotion
            
            # background bar
            cv2.rectangle(frame, (panel_x, y_pos), (panel_x + bar_width, y_pos + bar_height), (30, 30, 30), -1)
            cv2.rectangle(frame, (panel_x, y_pos), (panel_x + bar_width, y_pos + bar_height), (80, 80, 80), 1)
            
            # probability bar with color
            emotion_color = self.emotion_colors[emotion]
            if bar_length > 0:
                cv2.rectangle(frame, (panel_x + 2, y_pos + 2), 
                             (panel_x + bar_length - 2, y_pos + bar_height - 2), emotion_color, -1)
            
            # percentage text
            percentage_text = f"{prob:.1%}"
            if emotion == 'Sad':
                percentage_text += " (reduced)"
            elif emotion == 'Neutral':
                percentage_text += " (boosted)"
                
            cv2.putText(frame, emotion_text, (panel_x + bar_width + 15, y_pos + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, percentage_text, (panel_x + bar_width + 120, y_pos + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, emotion_color, 2)
    
    def draw_performance_info(self, frame):
        """draw performance information"""
        # calculate fps
        current_fps = len(self.fps_history)
        avg_fps = np.mean(self.fps_history) if self.fps_history else 0
        
        # draw controls panel (bottom)
        self._draw_controls_panel(frame)
    
    def _draw_controls_panel(self, frame):
        """draw controls panel at bottom"""
        panel_y = frame.shape[0] - 80
        panel_height = 70
        
        # controls background
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, panel_y), (frame.shape[1], frame.shape[0]), (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.9, frame, 0.1, 0, frame)
        
        # border line
        cv2.line(frame, (0, panel_y), (frame.shape[1], panel_y), (100, 150, 255), 2)
        
        # control instructions
        controls = [
            ("'S'", "save screenshot", (100, 255, 100)),
            ("'Q'", "quit application", (255, 100, 100))
        ]
        
        # calculate centered positions for controls
        total_width = 400  # estimated width for both controls
        start_x = (frame.shape[1] - total_width) // 2
        x_positions = [start_x, start_x + 200]
        
        for i, ((key, desc, color), x_pos) in enumerate(zip(controls, x_positions)):
            if x_pos < frame.shape[1] - 150:  # ensure it fits
                cv2.putText(frame, key, (x_pos, panel_y + 25), 
                           cv2.FONT_HERSHEY_DUPLEX, 0.6, color, 2)
                cv2.putText(frame, desc, (x_pos, panel_y + 45), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # right-align the signature
        signature = "ai emotion detection v3.0"
        text_size = cv2.getTextSize(signature, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
        signature_x = frame.shape[1] - text_size[0] - 20  # 20px margin from right edge
        cv2.putText(frame, signature, (signature_x, panel_y + 65), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
    
    def run_live_detection(self):
        """run live emotion detection"""
        # startup banner
        print("\n" + "="*80)
        print("facial emotion detection system".center(80))
        print("="*80)
        print("initializing ai-powered emotion recognition...")
        print("base accuracy: 74.43% | some bias correction added")
        print("-"*80)
        print("controls:")
        print("   press 's' -> save screenshot")
        print("   press 'q' -> quit application")
        print("   window automatically opens in optimal size")
        print("-"*80)
        print("starting camera feed... please wait...")
        print("="*80 + "\n")
        
        # try multiple camera initialization methods
        cap = None
        camera_backends = [
            (cv2.CAP_DSHOW, "directshow (recommended for windows)"),
            (cv2.CAP_MSMF, "media foundation"),
            (cv2.CAP_ANY, "auto-detect")
        ]
        for backend, name in camera_backends:
            print(f"trying camera backend: {name}")
            try:
                cap = cv2.VideoCapture(0, backend)
                if cap.isOpened():
                    # test if we can actually read a frame
                    ret, test_frame = cap.read()
                    if ret and test_frame is not None:
                        print(f"camera initialized successfully with {name}")
                        break
                    else:
                        print(f"camera opened but cannot read frames with {name}")
                        cap.release()
                        cap = None
                else:
                    print(f"failed to open camera with {name}")
                    if cap:
                        cap.release()
                    cap = None
            except Exception as e:
                print(f"error with {name}: {e}")
                if cap:
                    cap.release()
                cap = None
        
        if cap is None or not cap.isOpened():
            print("\n" + "="*60)
            print("camera initialization failed!")
            print("="*60)
            print("troubleshooting steps:")
            print("1. make sure your camera is not being used by another app")
            print("2. check if camera is enabled in windows privacy settings:")
            print("   settings > privacy & security > camera")
            print("3. try unplugging and reconnecting usb camera")
            print("4. restart the application")
            print("5. try running as administrator")
            print("="*60)
            input("press enter to close...")
            return
        
        # configure camera settings with error handling
        try:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 768)
            cap.set(cv2.CAP_PROP_FPS, 30)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # reduce buffer to minimize lag
        except Exception as e:
            print(f"warning: could not set camera properties: {e}")
        
        frame_start_time = time.time()
        screenshot_count = 0
        analysis_frame_counter = 0  # counter for when to analyze emotions
        consecutive_failures = 0  # track consecutive frame read failures
        
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                consecutive_failures += 1
                print(f"warning: failed to read frame (attempt {consecutive_failures})")
                
                if consecutive_failures >= 10:
                    print("\n" + "="*60)
                    print("camera connection lost!")
                    print("="*60)
                    print("possible causes:")
                    print("- camera disconnected")
                    print("- another app took control of camera")
                    print("- camera driver issue")
                    print("- insufficient usb power")
                    print("="*60)
                    break
                
                # try to reconnect camera
                if consecutive_failures % 5 == 0:
                    print("attempting to reconnect camera...")
                    cap.release()
                    time.sleep(1)
                    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
                    if not cap.isOpened():
                        print("reconnection failed")
                    else:
                        print("camera reconnected")
                        consecutive_failures = 0
                
                time.sleep(0.1)  # small delay before retry
                continue
            
            # reset failure counter on successful read
            consecutive_failures = 0
            
            # frame rate limiting for 10 fps
            current_time = time.time()
            if current_time - frame_start_time < 1.0 / self.target_fps:
                continue
            
            self.frame_count += 1
            analysis_frame_counter += 1
            
            # detect faces
            faces = self.detect_faces(frame)
            
            # determine if we should analyze emotion this frame (every 3rd frame for more responsiveness)
            should_analyze = analysis_frame_counter >= 3
            if should_analyze:
                analysis_frame_counter = 0
            
            if len(faces) > 0:
                # use the first (largest) face
                face = faces[0]
                x, y, w, h = face
                
                # store current face coordinates for consistent drawing
                self.current_face_coords = (x, y, w, h)
                
                # ensure face region is valid and large enough
                if w > 30 and h > 30:
                    # add padding around face
                    padding = 10
                    x_start = max(0, x - padding)
                    y_start = max(0, y - padding)
                    x_end = min(frame.shape[1], x + w + padding)
                    y_end = min(frame.shape[0], y + h + padding)
                    
                    face_region = frame[y_start:y_end, x_start:x_end]
                    
                    if face_region.size > 0 and face_region.shape[0] > 10 and face_region.shape[1] > 10:
                        if should_analyze:
                            # predict emotion only on analysis frames
                            emotion_idx, confidence, probabilities = self.predict_emotion(face_region)
                            
                            # immediate updates for responsive detection
                            self.current_emotion_idx = emotion_idx
                            self.current_confidence = confidence
                            self.current_probabilities = probabilities
                        
                        # always draw emotion information using current face position
                        if self.current_emotion_idx is not None:
                            # both box and bars use same instant data
                            self.draw_emotion_info(frame, self.current_emotion_idx, self.current_confidence, 
                                                 self.current_probabilities, self.current_face_coords)
                        else:
                            # analyzing state
                            cv2.rectangle(frame, (x, y), (x + w, y + h), (128, 128, 128), 2)
                            cv2.putText(frame, "analyzing...", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 128, 128), 2)
                    else:
                        # reset emotion state for invalid face regions
                        self.current_emotion_idx = None
                        self.current_confidence = 0.0
                        self.current_probabilities = np.ones(7) / 7
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                        cv2.putText(frame, "face too small - move closer", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
                else:
                    # reset emotion state for small faces
                    self.current_emotion_idx = None
                    self.current_confidence = 0.0
                    self.current_probabilities = np.ones(7) / 7
                    self.current_face_coords = None
                    # draw small face indicators for all detected faces
                    for (fx, fy, fw, fh) in faces:
                        cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), (255, 0, 0), 2)
                    cv2.putText(frame, "face too small", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.putText(frame, "tips: good lighting, face center, move closer", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
            else:
                # reset emotion state when no face detected
                self.current_emotion_idx = None
                self.current_confidence = 0.0
                self.current_probabilities = np.ones(7) / 7
                self.current_face_coords = None
                
                cv2.putText(frame, "no face detected", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(frame, "tips: good lighting, face center, move closer", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
            
            # draw performance info
            self.draw_performance_info(frame)
            
            # update fps calculation
            if current_time - frame_start_time >= 1.0:
                # calculate actual fps for the last second
                actual_fps = self.frame_count / (current_time - frame_start_time + 1.0)
                self.fps_history.append(actual_fps)
                frame_start_time = current_time
                self.frame_count = 0  # reset frame count for next second
            
            # display frame
            window_title = "emotion detection v3.0 - 74.43% accuracy"
            cv2.imshow(window_title, frame)
            
            # handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                screenshot_count += 1
                screenshot_name = f"emotion_screenshot_{screenshot_count}.jpg"
                cv2.imwrite(screenshot_name, frame)
                print(f"screenshot saved: {screenshot_name}")
        
        cap.release()
        cv2.destroyAllWindows()
        print("camera closed. seeya!")

def main():
    parser = argparse.ArgumentParser(description='live emotion detection')
    parser.add_argument('--model', type=str, help='path to model file')
    args = parser.parse_args()
    
    detector = EmotionDetector(model_path=args.model)
    detector.run_live_detection()

if __name__ == "__main__":
    main()
