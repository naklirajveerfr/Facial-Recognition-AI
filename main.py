import os
import cv2
import torch
import time
import warnings
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
from facenet_pytorch import MTCNN, InceptionResnetV1
from transformers import ViTImageProcessor, ViTForImageClassification
import torchvision.transforms.functional as F_ts


warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

class FaceAIApp:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        self.window.geometry("900x650")
        self.window.configure(bg="#1e1e2e")

        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        
       
        self.mtcnn = MTCNN(image_size=160, margin=20, keep_all=False, device=self.device)
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        
        model_name = "dima806/facial_emotions_image_detection"
        self.processor = ViTImageProcessor.from_pretrained(model_name)
        self.emotion_model = ViTForImageClassification.from_pretrained(model_name).eval().to(self.device)

       
        self.load_database()

     
        self.video_capture = cv2.VideoCapture(0)
        self.video_capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        

        self.frame_count = 0
        self.ai_skip_rate = 5
        self.cached_results = []
        self.prev_time = time.time()

 
        self.header = tk.Label(self.window, text=f"ENGINE: {self.gpu_name}", font=("Arial", 12, "bold"), fg="#a6e3a1", bg="#313244", pady=10)
        self.header.pack(fill=tk.X)

        
        self.canvas = tk.Canvas(self.window, width=800, height=500, bg="#11111b", highlightthickness=0)
        self.canvas.pack(pady=15)

        
        self.btn_quit = tk.Button(self.window, text="TERMINATE SYSTEM", command=self.on_closing, font=("Arial", 10, "bold"), fg="white", bg="#f38ba8", activebackground="#f38ba8", padx=20, pady=8, bd=0)
        self.btn_quit.pack()

      
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)


        self.update_frame()

    def load_database(self):
        KNOWN_FACES_DIR = "known_faces"
        self.known_embeddings = []
        self.known_names = []
        if os.path.exists(KNOWN_FACES_DIR):
            for filename in os.listdir(KNOWN_FACES_DIR):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    path = os.path.join(KNOWN_FACES_DIR, filename)
                    img = Image.open(path).convert('RGB')
                    face = self.mtcnn(img)
                    if face is not None:
                        with torch.no_grad():
                            emb = self.resnet(face.unsqueeze(0).to(self.device)).detach()
                        self.known_embeddings.append(emb)
                        self.known_names.append(os.path.splitext(filename)[0].upper())

    def update_frame(self):
        ret, frame = self.video_capture.read()
        if ret:
            self.frame_count += 1
            
  
            if self.frame_count % self.ai_skip_rate == 0:
                current_ai_results = []
                small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
                rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                pil_small = Image.fromarray(rgb_small)

                with torch.no_grad():
                    boxes, _ = self.mtcnn.detect(pil_small)

                if boxes is not None:
                    rgb_full = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_full = Image.fromarray(rgb_full)

                    for box in boxes:
                        x1, y1, x2, y2 = [int(b * 2) for b in box]
                        x1, y1 = max(0, x1), max(0, y1)  #  THE FIX
                        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                        
                        face_crop = pil_full.crop((x1, y1, x2, y2))
                        
                        with torch.no_grad():
                            identity = "UNKNOWN"
                            if len(self.known_embeddings) > 0:
                                face_resized = face_crop.resize((160, 160))
                                face_tensor = F_ts.to_tensor(face_resized).to(self.device)
                                face_tensor = (face_tensor * 255.0 - 127.5) / 128.0
                                current_emb = self.resnet(face_tensor.unsqueeze(0)).detach()
                                
                                min_dist = 0.85  
                                for idx, known_emb in enumerate(self.known_embeddings):
                                    dist = torch.dist(current_emb, known_emb).item()
                                    if dist < min_dist:
                                        min_dist = dist
                                        identity = self.known_names[idx]

                            inputs = self.processor(images=face_crop, return_tensors="pt").to(self.device)
                            outputs = self.emotion_model(**inputs)
                            top_idx = torch.argmax(outputs.logits, dim=-1).item()
                            emotion_label = self.emotion_model.config.id2label[top_idx].upper()

                        current_ai_results.append({
                            'box': (x1, y1, x2, y2),
                            'text': f"{identity} | {emotion_label}"
                        })
                self.cached_results = current_ai_results

    
            for face in self.cached_results:
                x1, y1, x2, y2 = face['box']
                theme_color = (0, 255, 0) if "UNKNOWN" not in face['text'] else (0, 165, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), theme_color, 2)
                cv2.rectangle(frame, (x1, y1 - 30), (x2, y1), theme_color, -1)
                cv2.putText(frame, face['text'], (x1 + 6, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

 
            current_time = time.time()
            fps = 1 / (current_time - self.prev_time) if (current_time - self.prev_time) > 0 else 0
            self.prev_time = current_time
            cv2.putText(frame, f"FPS: {int(fps)}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)


            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(rgb_frame))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

        self.window.after(10, self.update_frame)

    def on_closing(self):

        if messagebox.askokcancel("Quit", "Do you want to safely stop the production pipeline?"):
            self.video_capture.release()
            self.window.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = FaceAIApp(root, "Enterprise Biometric Platform v1.0")
    root.mainloop()
