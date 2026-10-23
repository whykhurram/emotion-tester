# 🙂 Facial Expression & Emotion Detection (Streamlit + OpenCV + Gemini)

## 1. Project overview
A beginner-friendly web app. You take a photo with your camera, the app finds your face, and (optionally) asks Google Gemini Flash to estimate your facial expression as Happy, Sad, Angry, Surprised, Fearful, Disgusted or Neutral.

> **Disclaimer:** the result is an AI estimate of a *facial expression*, not a reliable measurement of anyone's actual emotional or mental state.

## 2. Features
- Camera capture in the browser (`st.camera_input`) plus photo-upload fallback
- Local face detection with OpenCV (no data leaves the server for this step)
- Handles: no face, multiple faces (you pick one), camera unavailable, missing API key, unreadable images
- Optional **Analyze with Gemini** button: sends only the cropped face of one photo
- Emotion label + model-reported confidence + one-sentence explanation
- API key kept in Streamlit secrets / environment variable, never in code

## 3. Project structure
```
your-project/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore                  (you create this, see section 13)
└── .streamlit/secrets.toml     (local only, never committed)
```

## 4. Requirements
- Python 3.10 – 3.12 (3.13 should also work)
- A free Gemini API key (only needed for emotion classification)
- A modern browser with a camera (Chrome, Edge, Firefox, Safari)

## 5. Installation & virtual environment
```bash
cd your-project

# create
python -m venv .venv

# activate: macOS / Linux
source .venv/bin/activate
# activate: Windows PowerShell
.venv\Scripts\Activate.ps1
# activate: Windows cmd
.venv\Scripts\activate.bat
```

## 6. Install dependencies
```bash
pip install -r requirements.txt
```

## 7. Get and configure the Gemini API key
1. Go to <https://aistudio.google.com/apikey> and sign in with a Google account.
2. Click **Create API key** and copy it.
3. Configure it locally with **one** of these:

**Option A – Streamlit secrets (recommended).** Create `.streamlit/secrets.toml`:
```toml
GEMINI_API_KEY = "paste-your-key-here"
```

**Option B – environment variable.**
```bash
# macOS / Linux
export GEMINI_API_KEY="paste-your-key-here"
# Windows PowerShell
$env:GEMINI_API_KEY="paste-your-key-here"
```

Optional: set `GEMINI_MODEL` (environment variable) if you want a different Flash model. The default is `gemini-2.5-flash`.

## 8. Run locally
```bash
streamlit run app.py
```
Open the URL shown (usually <http://localhost:8501>) and allow camera access.

## 9. How the pipeline works
1. **Capture** – `st.camera_input` returns one still photo each time you press the shutter.
2. **Detect** – OpenCV Haar cascades find faces in that photo, locally.
3. **Select** – if several faces are found, you choose one; the largest is the default.
4. **Analyze (optional)** – when you press the button, the cropped face is sent to Gemini Flash, which returns JSON (`emotion`, `confidence`, `explanation`) that the app validates.
5. **Display** – label, confidence bar and explanation.

**Why not true real-time?** Streamlit re-runs the whole script on every interaction and its camera widget only delivers still photos, not a video stream. Real-time video would need a WebRTC component (e.g. `streamlit-webrtc`), which is harder to deploy (it often needs TURN servers on cloud hosts). Even then, calling Gemini on every frame would be slow, costly and rate-limited. The practical design here is *capture a frame, then analyze that one frame on demand*.

**Why Gemini for the emotion step?** A local expression classifier needs a model file (e.g. MediaPipe Face Landmarker or an FER network) that would be an extra download beyond the three project files. OpenCV does the cheap local work (face detection, smile check); Gemini does the classification only when asked.

## 10. Privacy considerations
- The app does not store photos; they live in memory during your session.
- Face detection is local. Pressing **Analyze with Gemini** sends the cropped face to Google. Read the current Gemini API terms to see how API data may be handled for your plan.
- Get consent from anyone you photograph. Do not use this for hiring, grading, surveillance or other decisions about people.
- Never commit your API key.

## 11. Limitations
- Facial expression ≠ inner emotion. People smile when sad and look neutral when upset.
- Accuracy varies with lighting, angle, glasses, masks, image quality, culture, age, and individual differences; bias across groups is a known issue.
- The confidence value is the model's self-report, not a calibrated probability.
- Haar cascades work best with frontal, well-lit faces and can miss or falsely detect faces.
- Results can differ between runs.

## 12. Create a GitHub repository
1. Sign in at <https://github.com> and click **+ → New repository**.
2. Name it (e.g. `emotion-detector`). Choose Public (needed for free Streamlit Cloud unless you link a private repo).
3. Do **not** tick "Add a README" (you already have one). Click **Create repository**.
4. Copy the repository URL shown, e.g. `https://github.com/YOUR_USERNAME/emotion-detector.git`.

## 13. Push the project to GitHub
Create a `.gitignore` **before** `git add .` so your secret key is never uploaded:
```bash
printf ".venv/\n__pycache__/\n.streamlit/secrets.toml\n" > .gitignore
```
Then:
```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin YOUR_GITHUB_REPOSITORY_URL
git push -u origin main
```
Replace `YOUR_GITHUB_REPOSITORY_URL` with the URL you copied in section 12. If asked for a password, use a GitHub Personal Access Token (or sign in via GitHub Desktop / `gh auth login`).

## 14. Deploy on Streamlit Community Cloud
1. Go to <https://share.streamlit.io> and sign in with GitHub; authorize access to your repositories.
2. Click **Create app** → deploy from an existing repo.
3. Choose your repository, branch `main`, and **Main file path** `app.py`.
4. Open **Advanced settings → Secrets** and add the key (section 15).
5. Click **Deploy**. The first build takes a few minutes.

## 15. Add `GEMINI_API_KEY` to Streamlit Cloud Secrets
In **Advanced settings → Secrets** (or later: app menu **⋮ → Settings → Secrets**) paste:
```toml
GEMINI_API_KEY = "paste-your-key-here"
```
Save. The app restarts and reads the key via `st.secrets`.

**Test the deployed camera:** open the app URL (HTTPS), choose **Camera**, allow the browser permission prompt, take a photo, confirm the face box appears, then press **Analyze with Gemini**.

## 16. Troubleshooting
| Problem | Fix |
|---|---|
| Camera does not start | Allow camera permission in the browser (lock icon in address bar); close other apps using the camera; use HTTPS or `localhost`; or use **Upload a photo**. |
| "No GEMINI_API_KEY found" | Add it to `.streamlit/secrets.toml` locally or to Cloud Secrets; restart the app. |
| "Gemini rejected the request" | Key is wrong, revoked, or lacks API access. Create a new key. |
| "Model not found" | Set `GEMINI_MODEL` to a currently available Flash model name. |
| "Rate limit or quota reached" | Wait a minute or check your quota in Google AI Studio. |
| "No face detected" | Better lighting, face the camera, move closer, remove obstructions. |
| `ModuleNotFoundError` | Activate the virtual environment and rerun `pip install -r requirements.txt`. |
| Cloud build fails on OpenCV (`libGL` error) | Make sure `requirements.txt` uses `opencv-python-headless`, not `opencv-python`. |
| Secrets not picked up on Cloud | Check the TOML syntax (key name exact, value in quotes) and reboot the app. |
