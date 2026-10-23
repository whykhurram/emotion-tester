"""
Facial Expression & Emotion Detection - Streamlit app

Pipeline
1. The user captures a photo with st.camera_input (or uploads one).
2. OpenCV (Haar cascades, runs locally) detects faces in that single photo.
3. Optional: the user presses "Analyze with Gemini" and ONLY the cropped face
   from that one photo is sent to Gemini Flash, which returns a structured
   emotion label + confidence. Nothing is sent automatically.

The result is an AI estimate of a facial expression, not a measurement of a
person's real emotional or mental state.
"""

import hashlib
import io
import os
from enum import Enum

import cv2
import numpy as np
import streamlit as st
from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel

# Model can be changed without editing code: set GEMINI_MODEL in env/secrets.
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
MAX_SIDE = 1024  # downscale large photos so processing stays fast


# --------------------------------------------------------------------------
# Data model for Gemini's structured response
# --------------------------------------------------------------------------
class Emotion(str, Enum):
    HAPPY = "Happy"
    SAD = "Sad"
    ANGRY = "Angry"
    SURPRISED = "Surprised"
    FEARFUL = "Fearful"
    DISGUSTED = "Disgusted"
    NEUTRAL = "Neutral"


class EmotionResult(BaseModel):
    emotion: Emotion
    confidence: float  # 0.0 - 1.0, the model's own (uncalibrated) estimate
    explanation: str   # one short sentence about visible facial cues


EMOJI = {
    "Happy": "😄", "Sad": "😢", "Angry": "😠", "Surprised": "😲",
    "Fearful": "😨", "Disgusted": "🤢", "Neutral": "😐",
}

PROMPT = (
    "You are looking at a cropped photo of ONE human face. Classify the visible "
    "facial expression as exactly one of: Happy, Sad, Angry, Surprised, "
    "Fearful, Disgusted, Neutral. Base your answer only on visible facial "
    "cues (mouth, eyebrows, eyes), not on guesses about the person's real "
    "feelings. Give a confidence between 0.0 and 1.0 and one short sentence "
    "describing the cues you used."
)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def get_api_key():
    """Read GEMINI_API_KEY from Streamlit secrets, then environment variables."""
    try:
        key = st.secrets.get("GEMINI_API_KEY")
    except Exception:  # no secrets file present
        key = None
    key = key or os.getenv("GEMINI_API_KEY") or ""
    return key.strip() or None


def load_image(data: bytes) -> Image.Image:
    """Decode bytes into an upright RGB PIL image, downscaled if very large."""
    img = Image.open(io.BytesIO(data))
    img = ImageOps.exif_transpose(img).convert("RGB")
    img.thumbnail((MAX_SIDE, MAX_SIDE))
    return img


@st.cache_resource
def load_cascades():
    """Load OpenCV's bundled Haar cascades once per server process."""
    face = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    smile = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")
    if face.empty() or smile.empty():
        raise RuntimeError("Could not load OpenCV cascade files.")
    return face, smile


def detect_faces(rgb: np.ndarray):
    """Return face boxes (x, y, w, h), largest first."""
    face_cascade, _ = load_cascades()
    gray = cv2.equalizeHist(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))
    found = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=6, minSize=(60, 60)
    )
    boxes = [tuple(int(v) for v in f) for f in found]
    return sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)


def draw_boxes(rgb: np.ndarray, faces, selected: int) -> np.ndarray:
    """Draw numbered boxes; the selected face is highlighted."""
    out = rgb.copy()
    for i, (x, y, w, h) in enumerate(faces):
        color = (46, 204, 113) if i == selected else (200, 200, 200)
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 3)
        cv2.putText(out, str(i + 1), (x + 6, y + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)
    return out


def crop_face(img: Image.Image, box, pad: float = 0.25) -> Image.Image:
    """Crop a face with some padding so the whole expression is visible."""
    x, y, w, h = box
    px, py = int(w * pad), int(h * pad)
    left, top = max(x - px, 0), max(y - py, 0)
    right, bottom = min(x + w + px, img.width), min(y + h + py, img.height)
    return img.crop((left, top, right, bottom))


def local_smile_check(rgb: np.ndarray, box) -> bool:
    """Very rough local check: is a smile visible in the lower half of the face?"""
    _, smile_cascade = load_cascades()
    x, y, w, h = box
    gray = cv2.cvtColor(rgb[y + h // 2: y + h, x: x + w], cv2.COLOR_RGB2GRAY)
    if gray.size == 0:
        return False
    smiles = smile_cascade.detectMultiScale(
        gray, scaleFactor=1.7, minNeighbors=25, minSize=(25, 25)
    )
    return len(smiles) > 0


def analyze_with_gemini(api_key: str, face_img: Image.Image) -> EmotionResult:
    """Send ONE cropped face to Gemini and return a validated result."""
    buf = io.BytesIO()
    face_img.save(buf, format="JPEG", quality=90)

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=[
            types.Part.from_bytes(data=buf.getvalue(), mime_type="image/jpeg"),
            PROMPT,
        ],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=EmotionResult,
        ),
    )

    result = response.parsed
    if not isinstance(result, EmotionResult):
        if not response.text:
            raise ValueError("Gemini returned no content (the image may have been blocked).")
        result = EmotionResult.model_validate_json(response.text)
    result.confidence = min(max(float(result.confidence), 0.0), 1.0)
    return result


def explain_gemini_error(exc: Exception) -> str:
    """Turn API exceptions into beginner-friendly messages."""
    if isinstance(exc, genai_errors.APIError):
        code = getattr(exc, "code", None)
        if code in (400, 401, 403):
            return ("Gemini rejected the request. Check that your GEMINI_API_KEY is valid "
                    "and has access to the Gemini API.")
        if code == 404:
            return f"Model '{MODEL_NAME}' was not found. Set GEMINI_MODEL to a currently available Flash model."
        if code == 429:
            return "Gemini rate limit or quota reached. Wait a moment and try again."
        return f"Gemini API error ({code}): {getattr(exc, 'message', exc)}"
    return f"Could not analyze the image: {exc}"


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Emotion Detector", page_icon="🙂", layout="centered")

    st.title("🙂 Facial Expression Detector")
    st.caption("Take a photo, find the face, and get an AI estimate of the expression.")
    st.info(
        "⚠️ **Disclaimer:** This is an AI estimate of a *facial expression*. It is not a "
        "reliable measurement of a person's actual emotional or mental state."
    )

    api_key = get_api_key()

    with st.sidebar:
        st.header("Privacy")
        st.write(
            "- Photos are processed in memory and **not saved** by this app.\n"
            "- Face detection runs locally on the server.\n"
            "- Only when you press **Analyze with Gemini**, the cropped face from that "
            "one photo is sent to Google's Gemini API.\n"
            "- Only photograph people who agree to it."
        )
        st.header("Status")
        if api_key:
            st.success("Gemini API key found")
        else:
            st.warning("No GEMINI_API_KEY found. Face detection works; Gemini analysis is disabled.")

    source = st.radio("Image source", ["📷 Camera", "🖼️ Upload a photo"], horizontal=True)
    if source.startswith("📷"):
        shot = st.camera_input("Take a photo")
        st.caption("If the camera does not start, allow camera access in your browser "
                   "or switch to **Upload a photo**.")
    else:
        shot = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

    if shot is None:
        st.stop()

    # ---- Decode image ----
    data = shot.getvalue()
    try:
        img = load_image(data)
    except (UnidentifiedImageError, OSError, ValueError):
        st.error("The image could not be processed. Please try another photo.")
        st.stop()
    rgb = np.array(img)

    # ---- Detect faces ----
    try:
        faces = detect_faces(rgb)
    except Exception as exc:
        st.error(f"Face detection failed: {exc}")
        st.stop()

    if not faces:
        st.warning("No face detected. Face the camera, use good lighting, and keep your face fully in frame.")
        st.image(img, use_container_width=True)
        st.stop()

    selected = 0
    if len(faces) > 1:
        st.warning(f"{len(faces)} faces detected. Choose which one to analyze.")
        selected = st.selectbox(
            "Face to analyze", range(len(faces)),
            format_func=lambda i: f"Face {i + 1}" + (" (largest)" if i == 0 else ""),
        )

    face_img = crop_face(img, faces[selected])
    col1, col2 = st.columns([3, 2])
    with col1:
        st.image(draw_boxes(rgb, faces, selected), caption="Detected face(s)", use_container_width=True)
    with col2:
        st.image(face_img, caption="Face sent for analysis", use_container_width=True)
        smiling = local_smile_check(rgb, faces[selected])
        st.metric("Local smile check", "🙂 Smile seen" if smiling else "No smile seen")
        st.caption("Rough local check only (OpenCV). Not an emotion classifier.")

    # ---- Optional Gemini analysis (only on button press) ----
    result_key = hashlib.sha256(data).hexdigest() + f":{selected}"
    if st.button("✨ Analyze with Gemini", type="primary", disabled=api_key is None):
        with st.spinner("Asking Gemini..."):
            try:
                result = analyze_with_gemini(api_key, face_img)
                st.session_state["result"] = (result_key, result)
            except Exception as exc:
                st.session_state.pop("result", None)
                st.error(explain_gemini_error(exc))

    stored = st.session_state.get("result")
    if stored and stored[0] == result_key:
        result = stored[1]
        label = result.emotion.value
        st.subheader(f"{EMOJI.get(label, '')} {label}")
        st.progress(result.confidence, text=f"Model-reported confidence: {result.confidence:.0%}")
        st.write(result.explanation)
        st.caption("Confidence is the model's own estimate, not a calibrated probability.")
    elif api_key is None:
        st.caption("Add GEMINI_API_KEY (see README) to enable emotion classification.")


if __name__ == "__main__":
    main()