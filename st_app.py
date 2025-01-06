import cv2
import mediapipe as mp
import streamlit as st

# Set page configuration
st.set_page_config(
    page_title="Yoga Pose Detection",
    page_icon="🧘‍♀️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ========== UI Header Section ==========
# Logo and header
st.image("assets/images/logo.svg", width=70)
st.write("")
col1, col2, col3 = st.columns([1, 2, 1])  # Center-align header elements

with col2:
    st.markdown("<h1 style='text-align: center; margin-bottom: 0;'>BALANCE</h1>", unsafe_allow_html=True)
    st.markdown(
        "<h3 style='text-align: center; font-size: 20px; color: gray; margin-top: -20px;'>yoga in perfection</h3>",
        unsafe_allow_html=True
    )
    st.markdown("<div style='height: 50px;'></div>", unsafe_allow_html=True)
    st.markdown(
        """
        <div style='text-align: center; font-size: 18px; color: #555; line-height: 2; margin: 0 0 50px;'>
        Discover the perfect balance in your yoga practice with our real-time guidance system. 
        Like having a personal yoga instructor, we provide instant feedback on your pose alignment 
        and symmetry. Whether you're mastering the basics or refining advanced poses, 
        we're here to support your journey to better form and deeper practice. 
        Start your enhanced yoga experience today!        
        </div>
        """,
        unsafe_allow_html=True
    )

# ========== Pose Detection Initialization ==========
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# ========== Helper Functions ==========
def resize_with_aspect_ratio(frame, target_width=None, target_height=None):
    """Resize a frame while maintaining aspect ratio."""
    height, width = frame.shape[:2]
    if target_width is not None:
        new_height = int((target_width / width) * height)
        return cv2.resize(frame, (target_width, new_height))
    if target_height is not None:
        new_width = int((target_height / height) * width)
        return cv2.resize(frame, (new_width, target_height))
    return frame

def calculate_distance(point1, point2):
    """Calculate Euclidean distance between two points."""
    return ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) ** 0.5

def process_frame(frame, pose):
    """Extract pose landmarks from a frame."""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        return [(lm.x, lm.y) for lm in landmarks], results
    return None, None

def compare_movements(joints1, joints2, threshold=0.05):
    """Compare movements to check symmetry."""
    if not joints1 or not joints2:
        return False, "No joints detected"
    left_elbow1, right_elbow1 = joints1[13], joints1[14]
    left_wrist1, right_wrist1 = joints1[15], joints1[16]
    left_elbow2, right_elbow2 = joints2[13], joints2[14]
    left_wrist2, right_wrist2 = joints2[15], joints2[16]
    distance1 = calculate_distance(left_elbow1, left_wrist1)
    distance2 = calculate_distance(right_elbow2, right_wrist2)
    if abs(distance1 - distance2) > threshold:
        return False, "Arms not symmetrical"
    return True, "Symmetry achieved"

# ========== Video Selection and Upload Section ==========
video_options = {
    "Warrior Pose": "assets/yoga_videos/warrior1.mp4",
    "Warrior II Pose": "assets/yoga_videos/warrior2.mp4",
    "Tree Pose": "assets/yoga_videos/tree_pose.mp4",
    "Downward Dog": "assets/yoga_videos/downward_dog.mp4"
}

# Centered video selection dropdown
_, col2, _ = st.columns([1, 3, 1])
with col2:
    selected_video = st.selectbox(
        "**🎥 Drag through our yoga pose library and let's practice together!**",
        list(video_options.keys()),
        format_func=lambda x: f"🧘 {x}",
    )
video_path = video_options[selected_video]

# Centered file uploader
st.write("")
_, col2, _ = st.columns([1, 3, 1])
with col2:
    uploaded_video = st.file_uploader(
        "**✨ Got your own favorite yoga video? Feel free to upload it here!**",
        type=["mp4"],
    )

if uploaded_video:
    with open("uploaded_video.mp4", "wb") as f:
        f.write(uploaded_video.read())
    video_path = "uploaded_video.mp4"

# Open reference video
reference_video = cv2.VideoCapture(video_path)
if not reference_video.isOpened():
    st.error(f"Unable to load the video: {video_path}. Check the file path!")

# ========== Slider Section ==========
st.write("")
_, col2, _ = st.columns([1, 3, 1])
with col2:
    st.markdown("**🎯 Fine-Tune Symmetry Check**")
    st.write(
        "<p style='font-size: 12px; color: gray;'>Adjust the threshold to control the sensitivity of the symmetry check. Lower values are stricter, while higher values allow more tolerance.</p>", 
        unsafe_allow_html=True
    )
    threshold = st.slider(
        "",  # Keep the slider without a duplicate label
        0.01, 
        0.1, 
        0.05
    )

# ========== Webcam and Video Display Section ==========
st.markdown("<div style='height: 50px;'></div>", unsafe_allow_html=True)
col1, col2 = st.columns([1, 1])
webcam_placeholder = col1.empty()
video_placeholder = col2.empty()

# ========== Pose Detection Logic ==========
cap = cv2.VideoCapture(0)
results = []
frame_skip = 2
frame_count = 0

while cap.isOpened():
    ret_webcam, webcam_frame = cap.read()
    if not ret_webcam:
        st.error("Unable to access the webcam.")
        break

    frame_count += 1
    if frame_count % frame_skip != 0:
        continue

    ret_ref, ref_frame = reference_video.read()
    if not ret_ref:
        reference_video.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    # Process frames
    webcam_frame_resized = resize_with_aspect_ratio(webcam_frame, target_width=640)
    ref_frame_resized = resize_with_aspect_ratio(ref_frame, target_width=640)
    webcam_joints, webcam_results = process_frame(webcam_frame_resized, pose)
    ref_joints, _ = process_frame(ref_frame_resized, pose) if ret_ref else (None, None)

    if webcam_results:
        mp_drawing.draw_landmarks(
            webcam_frame_resized,
            webcam_results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )

    is_symmetrical, feedback = compare_movements(webcam_joints, ref_joints, threshold=threshold) if ref_joints else (True, "Symmetry achieved")
    color = (0, 255, 0) if is_symmetrical else (0, 0, 255)
    cv2.putText(webcam_frame_resized, feedback, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # Update placeholders
    webcam_placeholder.image(cv2.cvtColor(webcam_frame_resized, cv2.COLOR_BGR2RGB), channels="RGB")
    video_placeholder.image(cv2.cvtColor(ref_frame_resized, cv2.COLOR_BGR2RGB), channels="RGB")
    results.append({"frame": frame_count, "symmetrical": is_symmetrical})

# Release resources
cap.release()
reference_video.release()
