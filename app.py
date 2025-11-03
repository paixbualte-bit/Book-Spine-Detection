import streamlit as st
import cv2
import av
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoSourceCallback

# Set page configuration
st.set_page_config(page_title="Book Spine Detection", layout="wide")
st.title("📚 Real-Time Library Book Spine Detection")
st.write("Show your webcam to the app and it will detect book spines.")

# Load your trained YOLOv8 model
# Make sure 'best.pt' is in the same folder as this script
try:
    model = YOLO("best.pt")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.error("Make sure 'best.pt' is in the same directory.")
    st.stop()

# Define the callback function for processing video frames
def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    # Convert the av.VideoFrame to a NumPy array (OpenCV format)
    img = frame.to_ndarray(format="bgr24")

    # Perform inference on the frame
    # We use stream=True for a more efficient generator
    results = model(img, stream=True, verbose=False) 

    # Loop through the results
    for r in results:
        boxes = r.boxes.xyxy.cpu().numpy()  # Get bounding boxes as numpy array
        confs = r.boxes.conf.cpu().numpy()  # Get confidences

        for box, conf in zip(boxes, confs):
            # Unpack coordinates
            x1, y1, x2, y2 = map(int, box)

            # Draw the bounding box on the image
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2) # Green box

            # Create label with confidence
            label = f"book_spine: {conf:.2f}"

            # Put the label above the box
            cv2.putText(img, label, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Convert the modified NumPy array back to an av.VideoFrame
    return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- Streamlit Web App Interface ---

# Add a note about permissions
st.info("ℹ️ You will need to grant webcam permissions in your browser.")

# Start the WebRTC streamer
webrtc_streamer(
    key="book-spine-detector",
    video_frame_callback=video_frame_callback,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
    rtc_configuration={  # This is needed for deployment
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    }
)

st.write("---")
st.write("Project by a beginner engineering student.")