# ASL Data Collection

A Streamlit proof of concept for collecting English sentence / ASL signing pairs and reviewing them with MediaPipe landmark overlays.

## Deploy to Streamlit Community Cloud

1. Commit `app.py`, `requirements.txt`, and `Random English Sentences.txt` to GitHub. The app selects a random non-empty line from the corpus.
2. Create a Community Cloud app using this repository and set the main file path to `app.py`. In **Advanced settings**, choose Python **3.12**; MediaPipe 0.10.21 publishes a Linux wheel for Python 3.12.
3. In the app's **Settings → Secrets** field, paste (replace the sample with a long unique password):

```toml
admin_password = "replace-with-a-long-random-password"
```

Do not commit a real `.streamlit/secrets.toml` file. The `.gitignore` excludes it.

Users can switch between the Deaf user and Developer review pages using the sidebar. Review access is protected by the configured password. No secrets are needed for sentence collection.

## Recording and storage notes

The browser camera requires HTTPS and camera permission. The app uses WebRTC to receive video frames, runs MediaPipe Holistic for pose, face, and both hands, and stores the source video and per-frame landmarks in process-wide in-memory queues. Reviewer overlays are rendered from the saved landmarks. `packages.txt` installs the Linux OpenCV runtime libraries required by the OpenCV build used with MediaPipe.

Records are in memory only. They disappear when the server restarts or sleeps, as expected for this proof of concept. They are shared across browser sessions while they use the same running app process, but are not durable and would not be shared across multiple server replicas. Streamlit Community Cloud needs WebRTC/STUN connectivity; some restrictive networks may block the live camera stream.

## Run locally (optional)

Install packages with `pip install -r requirements.txt`, then `streamlit run app.py`. Add a local `.streamlit/secrets.toml` with the same TOML entry to access the developer page. Do not commit it. Use Python 3.12 for local runs as well.
