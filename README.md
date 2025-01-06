# BALANCE - Yoga Pose Detection App 🧘‍♀️

A real-time yoga pose detection application that provides instant feedback on pose alignment and symmetry using computer vision.

## Features

- Real-time pose detection and tracking
- Side-by-side comparison with reference videos
- Instant feedback on pose alignment
- Support for custom video uploads
- Built-in library of yoga poses

## Prerequisites

- Python 3.11
- Docker (optional)
- Webcam

## Installation

### Using Docker

1. Build the Docker image:
```bash
docker build -t yoga_pose_app .
```

2. Run the container:
```bash
docker run -p 8501:8501 yoga_pose_app
```

3. Open your browser and navigate to:
```
http://localhost:8501
```

### Local Development

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run st_app.py
```

## Usage

1. Select a yoga pose from the built-in library
2. Allow webcam access when prompted
3. Follow the reference video and maintain the pose
4. Receive real-time feedback on your pose alignment

## Technologies Used

- Streamlit
- OpenCV
- MediaPipe
- Python 3.11
- Docker
