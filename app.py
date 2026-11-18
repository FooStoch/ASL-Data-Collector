from __future__ import annotations

import io
import json
import random
import tempfile
import threading
import time
import uuid
from pathlib import Path

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
        self.started = time.time()
        self.holistic = mp.solutions.holistic.Holistic(
            static_image_mode=False, model_complexity=1,
            enable_segmentation=False, refine_face_landmarks=True,
            min_detection_confidence=0.5, min_tracking_confidence=0.5,
        )

    @staticmethod
    def pack(points):
        if points is None:
            return None
        return [[round(p.x, 5), round(p.y, 5), round(p.z, 5), round(p.visibility, 4)]
                for p in points.landmark]

    def process(self, frame):
        image = frame.to_ndarray(format="bgr24")
        # Cap frame dimensions to keep CPU, memory, and in-memory video size manageable.
        h, w = image.shape[:2]
        scale = min(1.0, 720 / max(h, w))
        if scale < 1:
            image = cv2.resize(image, (int(w * scale), int(h * scale)))
        result = self.holistic.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        points = {
            "pose": self.pack(result.pose_landmarks),
            "face": self.pack(result.face_landmarks),
            "left_hand": self.pack(result.left_hand_landmarks),
            "right_hand": self.pack(result.right_hand_landmarks),
        }
        with self.lock:
            self.frames.append(image.copy())
            self.landmarks.append(points)
        return frame


def make_video(frames: list[np.ndarray]) -> bytes:
    if not frames:
        return b""
    height, width = frames[0].shape[:2]
    with tempfile.NamedTemporaryFile(suffix=".mp4") as tmp:
        writer = cv2.VideoWriter(tmp.name, cv2.VideoWriter_fourcc(*"mp4v"), 15.0, (width, height))
        if not writer.isOpened():
            raise RuntimeError("Could not initialize video encoder on this server.")
        for frame in frames:
            writer.write(frame)
        writer.release()
        tmp.seek(0)
        return tmp.read()


def overlay_video(frames: list[np.ndarray], landmarks: list[dict]) -> bytes:
    if not frames:
        return b""
    holistic = mp.solutions.holistic
    drawer = mp.solutions.drawing_utils
    specs = [
        ("pose", holistic.POSE_CONNECTIONS, (80, 220, 90)),
        ("face", mp.solutions.face_mesh.FACEMESH_TESSELATION, (210, 170, 70)),
        ("left_hand", holistic.HAND_CONNECTIONS, (255, 120, 40)),
        ("right_hand", holistic.HAND_CONNECTIONS, (40, 160, 255)),
    ]
    rendered = []
    for frame, coords in zip(frames, landmarks):
        out = frame.copy()
        for key, connections, color in specs:
            raw = coords.get(key)
            if not raw:
                continue
            lm = landmark_pb2.NormalizedLandmarkList(
                landmark=[landmark_pb2.NormalizedLandmark(
                    x=p[0], y=p[1], z=p[2], visibility=p[3]) for p in raw]
            )
            drawer.draw_landmarks(out, lm, connections,
                landmark_drawing_spec=drawer.DrawingSpec(color=color, thickness=1, circle_radius=1),
                connection_drawing_spec=drawer.DrawingSpec(color=color, thickness=1, circle_radius=1))
        rendered.append(out)
    return make_video(rendered)


@st.cache_resource
def get_store():
    # Shared across browser sessions on this process, cleared when the app process restarts.
    return {"pending": [], "approved": [], "lock": threading.RLock()}


def init_state():
    st.session_state.store = get_store()
    defaults = {"sentence": None, "capture": None, "capture_sentence": None}
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
    c1, c2 = st.columns([1, 3])
    with c1:
        if st.button("Choose another sentence", use_container_width=True):
            st.session_state.sentence = choose_sentence()
            st.rerun()
    st.markdown("### Record your signing")
    if st.session_state.capture is None:
        st.session_state.capture = Capture()
        st.session_state.capture_sentence = st.session_state.sentence
    capture = st.session_state.capture
    context = webrtc_streamer(
        key="asl-recorder", mode=WebRtcMode.SENDONLY,
        video_frame_callback=capture.process,
        media_stream_constraints={"video": True, "audio": False},
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        async_processing=True,
    )
    st.caption("Select START to open your camera. Sign the sentence, then select STOP. Your camera preview is shown to you; landmarks are not drawn on this page.")
    if not context.state.playing:
        with capture.lock:
            ready = len(capture.frames) > 0
        if ready and st.button("Finish and save recording", type="primary", use_container_width=True):
            with capture.lock:
                frames = capture.frames[:]
                landmarks = capture.landmarks[:]
            with st.spinner("Saving video and extracting landmarks…"):
                try:
                    video = make_video(frames)
                    item = {"id": uuid.uuid4().hex, "sentence": st.session_state.capture_sentence,
                            "video": video, "frames": frames, "landmarks": landmarks,
                            "created": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())}
                    st.session_state.store["pending"].append(item)
                    st.session_state.capture.holistic.close()
                    st.session_state.capture = None
                    st.session_state.capture_sentence = None
                    st.session_state.sentence = choose_sentence()
                    st.success("Saved for developer review. You can start the next sentence whenever you are ready.")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Could not save this recording: {exc}")
    elif st.button("Finish recording", type="primary", use_container_width=True):
        st.info("Select STOP in the camera controls first, then choose Finish and save recording.")


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
                if item["video"]:
                    st.video(item["video"], format="video/mp4", width="stretch")
                try:
                    st.video(overlay_video(item["frames"], item["landmarks"]), format="video/mp4", width="stretch")
                    st.caption("Landmark overlay: pose, face, left hand, and right hand")
                except Exception as exc:
                    st.warning(f"Overlay could not be rendered: {exc}")
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
                try:
                    st.video(overlay_video(item["frames"], item["landmarks"]), format="video/mp4", width="stretch")
                    st.caption("Approved video with landmark overlay")
                except Exception as exc:
                    st.warning(f"Overlay could not be rendered: {exc}")
                st.caption(f"Recorded {item['created']}")


init_state()
page = st.sidebar.radio("Page", ["Deaf user", "Developer review"], label_visibility="visible")
if page == "Deaf user":
    public_page()
else:
    admin_page()

