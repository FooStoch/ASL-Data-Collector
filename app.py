from __future__ import annotations

import io
import random
import threading
import time
import uuid
from fractions import Fraction
from pathlib import Path

import av
import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
from mediapipe.framework.formats import landmark_pb2
from streamlit_webrtc import WebRtcMode, webrtc_streamer

ROOT = Path(__file__).parent
CORPUS = ROOT / "Random English Sentences.txt"

st.set_page_config(page_title="ASL Data Collection", page_icon="🤟", layout="wide")
st.markdown("""
<style>
html, body, [class*="css"] { font-size: 18px; }
h1 { font-size: 2.35rem !important; }
.stButton > button { min-height: 3.1rem; font-size: 1.08rem; }
</style>
""", unsafe_allow_html=True)

@st.cache_data(show_spinner="Loading English sentences…")
def load_sentences() -> list[str]:
    # This corpus stores one sentence per line; ignore blank lines and a possible UTF-8 BOM.
    text = CORPUS.read_text(encoding="utf-8-sig")
    return [line.strip() for line in text.splitlines() if line.strip()]


def choose_sentence() -> str:
    sentences = load_sentences()
    previous = st.session_state.get("sentence")
    options = [s for s in sentences if s != previous]
    return random.choice(options or sentences)


class Capture:
    """Thread-safe frame and MediaPipe landmark buffer for one signing attempt."""
    def __init__(self):
        self.lock = threading.Lock()
        self.frames: list[np.ndarray] = []
        self.landmarks: list[dict] = []
        self.timestamps: list[float] = []
        self.started = time.time()
        self.holistic = mp.solutions.holistic.Holistic(
            static_image_mode=False, model_complexity=1,
            enable_segmentation=False, refine_face_landmarks=False,
            min_detection_confidence=0.5, min_tracking_confidence=0.5,
        )

    @staticmethod
    def pack(points):
        if points is None:
            return None
        return [[round(p.x, 5), round(p.y, 5), round(p.z, 5)]
                for p in points.landmark]

    def process(self, frame):
        # Preserve the camera/WebRTC timeline before landmark processing takes time.
        timestamp = frame.time
        if timestamp is None:
            timestamp = time.monotonic()
        image = frame.to_ndarray(format="bgr24")
        # Retain enough resolution for hand and lower-body landmarks while bounding memory.
        h, w = image.shape[:2]
        scale = min(1.0, 960 / max(h, w))
        if scale < 1:
            image = cv2.resize(image, (int(w * scale), int(h * scale)))
        # VP8 requires even frame dimensions.
        image = image[: image.shape[0] - image.shape[0] % 2,
                      : image.shape[1] - image.shape[1] % 2]
        result = self.holistic.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        points = {
            "pose": self.pack(result.pose_landmarks),
            "left_hand": self.pack(result.left_hand_landmarks),
            "right_hand": self.pack(result.right_hand_landmarks),
        }
        with self.lock:
            self.frames.append(image.copy())
            self.landmarks.append(points)
            self.timestamps.append(timestamp)
        return frame


def make_video(frames: list[np.ndarray], timestamps: list[float]) -> bytes:
    if not frames:
        return b""
    height, width = frames[0].shape[:2]
    output = io.BytesIO()
    with av.open(output, mode="w", format="webm") as container:
        # 30 fps is the nominal stream rate; explicit frame PTS values preserve the
        # camera's actual timing, including frames arriving at a different cadence.
        stream = container.add_stream("libvpx", rate=30)
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "30", "b": "0"}
        time_base = Fraction(1, 1000)
        stream.codec_context.time_base = time_base
        stream.time_base = time_base
        origin = timestamps[0] if timestamps else 0.0
        last_pts = -1
        for index, image in enumerate(frames):
            video_frame = av.VideoFrame.from_ndarray(image, format="bgr24")
            elapsed = timestamps[index] - origin if index < len(timestamps) else index / 30
            pts = max(last_pts + 1, round(max(0.0, elapsed) * 1000))
            video_frame.pts = pts
            video_frame.time_base = time_base
            last_pts = pts
            for packet in stream.encode(video_frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return output.getvalue()


def overlay_video(frames: list[np.ndarray], landmarks: list[dict], timestamps: list[float]) -> bytes:
    if not frames:
        return b""
    holistic = mp.solutions.holistic
    drawer = mp.solutions.drawing_utils
    specs = [
        # OpenCV draws BGR tuples: pose orange, left hand green, right hand magenta.
        ("pose", holistic.POSE_CONNECTIONS, (0, 145, 255), 2, 4),
        ("left_hand", holistic.HAND_CONNECTIONS, (0, 255, 0), 2, 6),
        ("right_hand", holistic.HAND_CONNECTIONS, (255, 0, 255), 2, 6),
    ]
    rendered = []
    for frame, coords in zip(frames, landmarks):
        out = frame.copy()
        for key, connections, color, line_width, dot_radius in specs:
            raw = coords.get(key)
            if not raw:
                continue
            # Leave visibility unset. Hand landmarks have no meaningful visibility
            # field; setting its default (0) causes MediaPipe drawing to hide them.
            lm = landmark_pb2.NormalizedLandmarkList(
                landmark=[landmark_pb2.NormalizedLandmark(x=p[0], y=p[1], z=p[2]) for p in raw]
            )
            drawer.draw_landmarks(out, lm, connections,
                landmark_drawing_spec=drawer.DrawingSpec(color=color, thickness=-1, circle_radius=dot_radius),
                connection_drawing_spec=drawer.DrawingSpec(color=color, thickness=line_width, circle_radius=dot_radius))
        rendered.append(out)
    return make_video(rendered, timestamps)


@st.cache_resource
def get_store():
    # Shared across browser sessions on this process, cleared when the app process restarts.
    return {"pending": [], "approved": [], "lock": threading.RLock()}


def init_state():
    st.session_state.store = get_store()
    defaults = {"sentence": None, "capture": None}
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def public_page():
    st.title("American Sign Language (ASL) Data Collection")
    st.write("Choose a sentence, then record yourself signing it. The camera view is used only to capture your signing; landmarks are saved with the recording.")
    if st.session_state.sentence is None:
        st.session_state.sentence = choose_sentence()
    st.markdown("### Sentence to sign")
    st.markdown(f"<div style='font-size:1.65rem; font-weight:650; padding:1rem; border-radius:0.7rem; background:#f0f2f6'>{st.session_state.sentence}</div>", unsafe_allow_html=True)
    if st.session_state.capture is None:
        st.session_state.capture = Capture()
    capture = st.session_state.capture
    with capture.lock:
        has_frames = bool(capture.frames)
    if st.button("Choose another sentence", use_container_width=True, disabled=has_frames):
        st.session_state.sentence = choose_sentence()
        st.rerun()
    st.markdown("### Record your signing")
    context = webrtc_streamer(
        key="asl-recorder", mode=WebRtcMode.SENDRECV,
        video_frame_callback=capture.process,
        media_stream_constraints={
            "video": {"width": {"ideal": 1280}, "height": {"ideal": 720}, "frameRate": {"ideal": 30}},
            "audio": False,
        },
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        async_processing=True,
    )
    st.caption("Select START to open your camera. For best landmarks, frame your full body and keep both hands visible. Your preview has no overlays. When you finish signing, select STOP, then save the recording.")
    if not context.state.playing:
        with capture.lock:
            ready = len(capture.frames) > 0
        if ready and st.button("Finish and save recording", type="primary", use_container_width=True):
            with capture.lock:
                frames = capture.frames[:]
                landmarks = capture.landmarks[:]
                timestamps = capture.timestamps[:]
            with st.spinner("Saving video and extracting landmarks…"):
                try:
                    # Save a single WebM clip with the landmark visualization composited on it.
                    video = overlay_video(frames, landmarks, timestamps)
                    item = {"id": uuid.uuid4().hex, "sentence": st.session_state.sentence,
                            "video": video, "landmarks": landmarks,
                            "created": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())}
                    st.session_state.store["pending"].append(item)
                    st.session_state.capture.holistic.close()
                    st.session_state.capture = None
                    st.session_state.sentence = choose_sentence()
                    st.success("Saved for developer review. You can start the next sentence whenever you are ready.")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Could not save this recording: {exc}")
    else:
        st.info("When you finish signing, select STOP in the camera controls. The save button will then appear.")


def admin_page():
    st.title("Developer review")
    try:
        expected = st.secrets["admin_password"]
    except (FileNotFoundError, KeyError):
        expected = ""
    if not expected:
        st.error("Developer access is not configured. Set admin_password in Streamlit app Secrets.")
        return
    if not st.session_state.get("admin_authenticated", False):
        with st.form("admin_login"):
            password = st.text_input("Developer password", type="password")
            submitted = st.form_submit_button("Unlock review queue")
        if submitted:
            if password == expected:
                st.session_state.admin_authenticated = True
                st.rerun()
            st.error("Incorrect password.")
        return
    if st.button("Lock developer page"):
        st.session_state.admin_authenticated = False
        st.rerun()
    left, right = st.columns(2, gap="large")
    with left:
        st.subheader(f"Needs review ({len(st.session_state.store['pending'])})")
        for item in list(st.session_state.store["pending"]):
            with st.container(border=True):
                st.markdown(f"**{item['sentence']}**")
                st.video(item["video"], format="video/webm", width="stretch")
                st.caption("Landmarks overlaid: pose, left hand, and right hand")
                good, reject = st.columns(2)
                with good:
                    if st.button("✓ Approve", key=f"approve_{item['id']}", use_container_width=True):
                        st.session_state.store["pending"].remove(item)
                        st.session_state.store["approved"].append(item)
                        st.rerun()
                with reject:
                    if st.button("✕ Reject", key=f"reject_{item['id']}", use_container_width=True):
                        st.session_state.store["pending"].remove(item)
                        st.rerun()
    with right:
        st.subheader(f"Approved pairs ({len(st.session_state.store['approved'])})")
        for item in st.session_state.store["approved"]:
            with st.container(border=True):
                st.markdown(f"**{item['sentence']}**")
                st.video(item["video"], format="video/webm", width="stretch")
                st.caption("Approved video with landmark overlay")
                st.caption(f"Recorded {item['created']}")


init_state()
page = st.sidebar.radio("Page", ["Deaf user", "Developer review"], label_visibility="visible")
if page == "Deaf user":
    public_page()
else:
    admin_page()

