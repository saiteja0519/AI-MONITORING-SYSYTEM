"""
app.py
------
Streamlit entrypoint for the Smart Meeting Attention Monitoring System.

Run with:
    streamlit run app.py

This module is intentionally thin: it wires together the detector pipeline,
attention engine, notifier, and database logger, and renders the live
dashboard + end-of-meeting report. All actual detection/scoring logic lives
in detector/, notifier/, database/, and reports/ so it can be unit-tested
and reused outside Streamlit (e.g. a future PyQt5 or CLI front-end).

IMPORTANT - consent & privacy:
This tool is designed for SELF-monitoring with the monitored person's own
consent. The webcam feed is processed locally/in-session only; nothing is
uploaded anywhere except the small text alert you explicitly configure to
send to your own phone via ntfy. See README.md "Privacy & Consent" section.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime

import cv2
import numpy as np
import streamlit as st

av = None
av_import_error = None
try:
    import av
except ImportError as exc:
    av_import_error = exc

webrtc_streamer = None
WebRtcMode = None
webrtc_import_error = None
try:
    from streamlit_webrtc import webrtc_streamer, WebRtcMode
except Exception as exc:
    webrtc_import_error = exc

DEPS_ERROR = None
if av_import_error is not None:
    DEPS_ERROR = (
        "Missing dependency 'av'. Install required packages with `pip install -r requirements.txt`. "
        f"({av_import_error})"
    )
elif webrtc_import_error is not None:
    DEPS_ERROR = (
        "Missing dependency for streamlit-webrtc. Install required packages with `pip install -r requirements.txt`. "
        f"({webrtc_import_error})"
    )

from config import CONFIG, DB_PATH
from detector.face_detector import FaceDetector
from detector.eye_detector import EyeDetector
from detector.head_pose import HeadPoseEstimator
from detector.gaze_tracker import GazeTracker
from detector.yawn_detector import YawnDetector
from detector.attention_score import AttentionEngine, OverallStatus
from notifier.popup import audio_autoplay_html, popup_banner_markdown
from notifier.mobile_notification import NtfyNotifier, ThrottledNotifier
from database.logger import SessionLogger
from reports.report_generator import build_report_data, generate_charts, export_csv, export_pdf

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("app")

st.set_page_config(page_title="Smart Meeting Attention Monitor", layout="wide")


# ---------------------------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------------------------
def init_state() -> None:
    face_detector = None
    face_detector_error = None
    try:
        face_detector = FaceDetector()
    except Exception as exc:
        logger.exception("FaceDetector initialization failed")
        face_detector_error = str(exc)

    defaults = {
        "monitoring": False,
        "consent_given": False,
        "session_logger": SessionLogger(DB_PATH),
        "db_session_id": None,
        "last_status": None,
        "last_alert_at": 0.0,
        "last_push_at": 0.0,
        "low_score_since": None,
        "current_frame_data": None,
        "show_alert_banner": False,
        "alert_sound_html": "",
        "notifier": ThrottledNotifier(NtfyNotifier(CONFIG.ntfy), CONFIG.alert.push_cooldown_seconds),
        "engine": AttentionEngine(CONFIG),
        "face_detector": face_detector,
        "face_detector_error": face_detector_error,
        "dependency_error": DEPS_ERROR,
        "eye_detector": EyeDetector(CONFIG.eye),
        "head_pose": HeadPoseEstimator(CONFIG.head_pose),
        "gaze_tracker": GazeTracker(CONFIG.gaze),
        "yawn_detector": YawnDetector(CONFIG.yawn),
        "heartbeat_last": 0.0,
        "meeting_label": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_state()


# ---------------------------------------------------------------------------
# Video frame callback (runs per-frame inside the webrtc worker thread)
# ---------------------------------------------------------------------------
def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    img = frame.to_ndarray(format="bgr24")
    now = time.time()

    if not st.session_state.monitoring or st.session_state.face_detector is None:
        return av.VideoFrame.from_ndarray(img, format="bgr24")

    face_result = st.session_state.face_detector.process(img)

    if not face_result.found:
        af = st.session_state.engine.evaluate(face_found=False, now=now)
    else:
        eye_res = st.session_state.eye_detector.update(face_result.landmarks_px, now=now)
        head_res = st.session_state.head_pose.update(
            face_result.landmarks_px, face_result.frame_width, face_result.frame_height, now=now
        )
        gaze_res = st.session_state.gaze_tracker.update(face_result.landmarks_px)
        yawn_res = st.session_state.yawn_detector.update(face_result.landmarks_px, now=now)
        af = st.session_state.engine.evaluate(
            face_found=True, eye=eye_res, head=head_res, gaze=gaze_res, yawn=yawn_res, now=now
        )

        # Draw a light face-mesh overlay for visual feedback
        for (x, y) in face_result.landmarks_px[::4].astype(int):
            cv2.circle(img, (x, y), 1, (0, 200, 0), -1)

    st.session_state.current_frame_data = af

    # --- status-change / heartbeat logging ---
    status_changed = st.session_state.last_status != af.status
    heartbeat_due = (now - st.session_state.heartbeat_last) >= 5.0
    if status_changed or heartbeat_due:
        st.session_state.session_logger.log_event(
            status=af.status.value,
            expression=af.expression.value,
            attention_score=af.score,
            reason=af.reason,
            ear=(af.eye.ear if af.eye else None),
            blink_count=(af.eye.blink_count if af.eye else None),
            yawn_count=(af.yawn.yawn_count if af.yawn else None),
        )
        st.session_state.heartbeat_last = now
    st.session_state.last_status = af.status

    # --- low-score sustained tracking -> warning + push ---
    cfg = CONFIG.alert
    is_low = af.score < cfg.low_score_threshold or af.status in (
        OverallStatus.NOT_PAYING_ATTENTION, OverallStatus.SLEEPING,
    )
    if is_low:
        if st.session_state.low_score_since is None:
            st.session_state.low_score_since = now
        sustained = now - st.session_state.low_score_since
    else:
        st.session_state.low_score_since = None
        sustained = 0.0

    if sustained >= cfg.sustained_seconds:
        if (now - st.session_state.last_alert_at) >= cfg.alert_cooldown_seconds:
            st.session_state.show_alert_banner = True
            st.session_state.last_alert_at = now
            st.session_state.session_logger.log_event(
                status=af.status.value, expression=af.expression.value,
                attention_score=af.score, reason=f"Warning triggered: {af.reason}",
            )
            if cfg.send_push:
                st.session_state.notifier.send(
                    "⚠️ Attention Alert",
                    f"You have not been paying attention for {int(sustained)}s. "
                    f"Please return to your meeting.",
                )

    # Overlay status text on the video frame itself
    color = (0, 200, 0) if af.status == OverallStatus.ATTENTIVE else (0, 0, 220)
    cv2.putText(img, f"{af.status.value} | Score: {af.score}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    return av.VideoFrame.from_ndarray(img, format="bgr24")


# ---------------------------------------------------------------------------
# Sidebar - consent, session controls, alert settings
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Session Controls")
    st.session_state.meeting_label = st.text_input("Meeting name (optional)", value=st.session_state.meeting_label)

    st.session_state.consent_given = st.checkbox(
        "I consent to my webcam being analyzed locally for self-monitoring "
        "purposes during this session.",
        value=st.session_state.consent_given,
    )

    if st.session_state.dependency_error is not None:
        st.error(st.session_state.dependency_error)
    elif st.session_state.face_detector_error is not None:
        st.error(
            "FaceDetector could not be initialized: "
            f"{st.session_state.face_detector_error}"
        )

    col_a, col_b = st.columns(2)
    with col_a:
        start_disabled = (
            st.session_state.monitoring
            or not st.session_state.consent_given
            or st.session_state.face_detector_error is not None
            or st.session_state.dependency_error is not None
        )
        if st.button("▶ Start Session", disabled=start_disabled, use_container_width=True):
            st.session_state.monitoring = True
            st.session_state.db_session_id = st.session_state.session_logger.start_session(
                st.session_state.meeting_label
            )
            st.session_state.engine.reset()
            st.session_state.eye_detector.reset()
            st.session_state.head_pose.reset()
            st.session_state.yawn_detector.reset()
            st.rerun()
    with col_b:
        if st.button("⏹ End Session", disabled=not st.session_state.monitoring, use_container_width=True):
            st.session_state.monitoring = False
            st.session_state.session_logger.end_session()
            st.rerun()

    st.divider()
    st.subheader("Alert Settings")
    CONFIG.alert.show_popup = st.checkbox("Show on-screen popup warning", value=CONFIG.alert.show_popup)
    CONFIG.alert.play_sound = st.checkbox("Play alert sound", value=CONFIG.alert.play_sound)
    CONFIG.alert.send_push = st.checkbox(
        "Send push notification to phone (via ntfy)", value=CONFIG.alert.send_push
    )
    if CONFIG.alert.send_push:
        CONFIG.ntfy.topic = st.text_input(
            "ntfy topic name", value=CONFIG.ntfy.topic,
            help="Subscribe to this same topic name in the ntfy mobile app.",
        )
        st.caption("Install the free ntfy app and subscribe to this topic to receive alerts.")

    CONFIG.alert.sustained_seconds = st.slider(
        "Seconds of low attention before warning", 3, 60, int(CONFIG.alert.sustained_seconds)
    )

    if not st.session_state.consent_given:
        st.info("Consent checkbox must be checked to start monitoring.")


# ---------------------------------------------------------------------------
# Main layout
# ---------------------------------------------------------------------------
st.title("🎯 Smart Meeting Attention Monitoring System")
st.caption("Self-monitoring tool for online meetings — processed locally, for personal use only.")

if st.session_state.show_alert_banner:
    if CONFIG.alert.show_popup:
        st.markdown(popup_banner_markdown("Please pay attention to your meeting."), unsafe_allow_html=True)
    if CONFIG.alert.play_sound:
        from config import ALERT_SOUND_PATH
        st.markdown(audio_autoplay_html(ALERT_SOUND_PATH), unsafe_allow_html=True)
    st.session_state.show_alert_banner = False  # one-shot per trigger

video_col, stats_col = st.columns([2, 1])

with video_col:
    if st.session_state.dependency_error is not None:
        st.error(
            "Cannot start webcam monitoring because required dependencies are missing. "
            "Install them with `pip install -r requirements.txt`."
        )
    else:
        webrtc_streamer(
            key="attention-monitor",
            mode=WebRtcMode.SENDRECV,
            video_frame_callback=video_frame_callback,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

with stats_col:
    st.subheader("Live Status")
    af = st.session_state.current_frame_data
    if af is None:
        st.info("Start a session and enable your webcam to begin monitoring.")
    else:
        st.metric("Attention Score", f"{af.score} / 100")
        st.metric("Status", af.status.value)
        st.metric("Expression", af.expression.value)
        if af.eye:
            st.metric("Blink Count", af.eye.blink_count)
        if af.yawn:
            st.metric("Yawn Count", af.yawn.yawn_count)
        st.caption(f"Reason: {af.reason}")

st.divider()

# ---------------------------------------------------------------------------
# End-of-meeting report section
# ---------------------------------------------------------------------------
st.header("📊 Session Reports")

sessions = st.session_state.session_logger.list_sessions()
if not sessions:
    st.caption("No sessions recorded yet.")
else:
    labels = [
        f"#{s['id']} - {s['meeting_label'] or 'Untitled'} - {s['started_at']}"
        for s in sessions
    ]
    selected_idx = st.selectbox("Select a session", range(len(sessions)), format_func=lambda i: labels[i])
    selected_session = sessions[selected_idx]
    logs = st.session_state.session_logger.get_logs_for_session(selected_session["id"])
    report_data = build_report_data(selected_session, logs)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Duration", f"{round(report_data.duration_seconds / 60, 1)} min")
    m2.metric("Avg Score", report_data.average_score)
    m3.metric("Max Distraction", f"{report_data.max_distraction_seconds}s")
    m4.metric("Warnings", report_data.total_warnings)

    if logs:
        charts = generate_charts(report_data)
        c1, c2 = st.columns(2)
        c1.image(charts["pie"], caption="Status Breakdown")
        c2.image(charts["bar"], caption="Event Counts")
        st.image(charts["line"], caption="Attention Timeline")

        from config import REPORTS_DIR
        pdf_path = REPORTS_DIR / f"session_{selected_session['id']}_report.pdf"
        csv_path = REPORTS_DIR / f"session_{selected_session['id']}_report.csv"

        dl1, dl2 = st.columns(2)
        with dl1:
            if st.button("Generate PDF Report"):
                export_pdf(report_data, pdf_path)
                st.success(f"PDF saved to {pdf_path}")
            if pdf_path.exists():
                with open(pdf_path, "rb") as f:
                    st.download_button("⬇ Download PDF", f, file_name=pdf_path.name)
        with dl2:
            export_csv(report_data, csv_path)
            with open(csv_path, "rb") as f:
                st.download_button("⬇ Download CSV", f, file_name=csv_path.name)
    else:
        st.caption("This session has no logged events yet.")

st.divider()
with st.expander("Privacy & Consent"):
    st.markdown(
        "- Your webcam video is processed **locally within this app session** "
        "for attention detection only.\n"
        "- No video/images are stored to disk or uploaded anywhere.\n"
        "- Only small text logs (status + numeric score, not images) are saved "
        "to a local SQLite database for the session report.\n"
        "- Mobile push notifications, if enabled, contain only a short text "
        "message and go through the ntfy service using a topic name you choose.\n"
        "- This tool is intended for **self-monitoring with your own consent**. "
        "Do not use it to monitor others without their knowledge and agreement."
    )
