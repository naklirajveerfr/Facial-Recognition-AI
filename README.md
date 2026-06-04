#  Facial Recognition And Sentimental Analysis

A high-performance, privacy-first, local computer vision application that provides real-time facial recognition and emotion analysis. Built with a robust **Tkinter GUI wrapper**, this system is engineered for edge production environments, ensuring zero cloud dependencies, secure localized data processing, and optimized resource management on lower-tier hardware.

---

##  Key Features

* **Real-Time Facial Identification:** Leverages a pre-trained **MTCNN** for high-accuracy face detection and an **InceptionResnetV1 (VGGFace2)** network for generating unique biometric embeddings.
* **Deep Emotion Analysis:** Integrates a state-of-the-art Hugging Face **Vision Transformer (ViT)** model (`dima806/facial_emotions_image_detection`) to classify facial expressions instantly.
* **Production-Grade Architecture:** Wrapped entirely in a native **Tkinter desktop GUI context**. Replaces volatile raw OpenCV streaming loops with asynchronous event scheduling to eliminate memory leaks and terminal hang-ups.
* **Low-VRAM Optimization Matrix:** Tailored specifically to maintain a stable ~30 FPS on entry-level dedicated graphics hardware (e.g., NVIDIA GTX 1050 4GB) via an automated dual-step optimization protocol:
    1.  *50% Downscaled Matrix Face-Sensing:* Minimizes the initial calculation overhead on the GPU.
    2.  *Calculated Execution Throttle:* Deep neural pipelines process inputs every 5th frame, while rendering overlays update natively at full stream speed.
* **Dynamic Hardware Adaptation:** Automated PyTorch driver querying natively identifies and binds to whatever GPU engine is executing the script, appending live hardware tracking metrics directly to the interface.



##  Installation & Setup

### 1. Project Directory Configuration
Clone or build your project folder structure exactly as formatted below to ensure absolute path integrity during initialization:

```text
C:\Lab\sfbs\
│
├── saai.py                  # Main Application Script
├── requirements.txt         # Package Dependency Manifest
└── known_faces/             # Face Registration Database 
    ├── Name1.jpg            # Clear, single-face profile image
    └── Name2.png            # Clear, single-face profile image
