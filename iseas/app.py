import customtkinter as ctk
import cv2
import face_recognition
import numpy as np
from ultralytics import YOLO
import requests
import threading
import logging
import winsound
import tempfile
import os
from concurrent.futures import ThreadPoolExecutor
from PIL import Image, ImageOps, ImageDraw
from datetime import datetime
import time
from typing import Optional, List, Tuple, Set, Dict
from scipy.spatial import distance as dist
import torch
import pyttsx3

# --- LOGGING SETUP ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- CONFIGURATION ---
SERVER_URL = "http://127.0.0.1:5000"
PPE_MODEL_PATH = 'best_.pt'
CONFIDENCE = 0.5

# ESP32 CONFIGURATION
ESP32_IP = "10.83.183.90"  # Change this to your ESP32's IP address
ESP32_PORT = 80
ESP32_URL = f"http://{ESP32_IP}:{ESP32_PORT}/gate"

# TIMING CONSTANTS
GRANT_DISPLAY_DURATION_SEC = 4
UNKNOWN_FACE_TIMEOUT_SEC = 3
LIVENESS_TIMEOUT_SEC = 8
SPOOF_DISPLAY_SEC = 3
FACE_SYNC_INTERVAL_MS = 60000
RULES_SYNC_INTERVAL_MS = 2000
API_TIMEOUT_SECONDS = 5
UPDATE_LOOP_DELAY_MS = 10

# RFID FALLBACK SETTINGS
RFID_POLL_INTERVAL_MS = 500  # Poll ESP32 for RFID every 500ms
RFID_TIMEOUT_SEC = 15  # Seconds to wait for RFID tap
RFID_CARD_NOT_FOUND_DISPLAY_SEC = 3  # How long to show "card not registered"

# LIVENESS DETECTION SETTINGS
EYE_AR_THRESH = 0.25  # Eye aspect ratio threshold for blink (higher = easier to trigger)
EYE_AR_CONSEC_FRAMES = 1  # Consecutive frames for blink detection
REQUIRED_BLINKS = 1  # Number of blinks required for liveness
LIVENESS_RESOLUTION = (640, 480)  # Higher resolution for liveness detection
EAR_DROP_RATIO = 0.70  # If EAR drops to 70% of running average, count as blink
MOTION_THRESHOLD = 15  # Pixel difference threshold for motion detection
MOTION_AREA_THRESHOLD = 500  # Minimum contour area for motion

# AUTO-BRIGHTNESS SETTINGS
TARGET_BRIGHTNESS_MIN = 80
TARGET_BRIGHTNESS_MAX = 170
BRIGHTNESS_CHECK_INTERVAL = 30  # Check every N frames
EXPOSURE_STEP = 5

# VOICE FEEDBACK SETTINGS
AUDIO_ENABLED = True
VOICE_RATE = 175  # Speech speed (words per minute)
VOICE_VOLUME = 1.0  # Volume (0.0 to 1.0)

# OPTIMIZATION SETTINGS
PROCESS_EVERY_N_FRAMES = 3
AI_RESOLUTION = (320, 240)
MAX_WORKERS = 4  # Increased from 2 — handles audio, logs, ESP32, sync

# CUDA / DEVICE DETECTION
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
logger.info(f"Using device: {DEVICE}" + (f" ({torch.cuda.get_device_name(0)})" if DEVICE == 'cuda' else ""))

# FACE MATCH TOLERANCE
FACE_DISTANCE_TOLERANCE = 0.6

PPE_NAMES = {
    0: "Person", 1: "Ear", 2: "Ear-muffs", 3: "Face",
    4: "Face-guard", 5: "Face-mask", 6: "Foot", 7: "Tool",
    8: "Glasses", 9: "Gloves", 10: "Helmet", 11: "Hands",
    12: "Head", 13: "Medical-suit", 14: "Shoes",
    15: "Safety-suit", 16: "Safety-vest"
}

# --- THEME COLORS ---
COLORS = {
    "bg_dark": "#0f1117",
    "bg_card": "#1a1d26",
    "bg_secondary": "#22262f",
    "border": "#2a2f3a",
    "text_primary": "#f0f2f5",
    "text_secondary": "#8b919e",
    "text_muted": "#5c6370",
    "accent_blue": "#3b82f6",
    "accent_green": "#10b981",
    "accent_red": "#ef4444",
    "accent_orange": "#f59e0b",
    "accent_purple": "#8b5cf6",
}

# --- STATUS VISUALS (lookup table instead of if/elif chain) ---
_STATUS_VISUALS = {
    "granted":   (COLORS["accent_green"],  "✅", "#0d3320"),
    "warning":   (COLORS["accent_orange"], "⚠️", "#3b2e0f"),
    "denied":    (COLORS["accent_red"],    "⛔", "#3b1515"),
    "unknown":   (COLORS["accent_red"],    "❓", "#3b1520"),
    "spoof":     (COLORS["accent_red"],    "🚫", "#3b1515"),
    "rfid":      (COLORS["accent_orange"], "💳", "#3b2e0f"),
    "rfid_fail": (COLORS["accent_red"],    "💳", "#3b1515"),
    "liveness":  (COLORS["accent_purple"], "👁️", "#2d1f4a"),
    "scanning":  (COLORS["text_secondary"], "🔍", COLORS["bg_secondary"]),
}
_STATUS_DEFAULT = (COLORS["text_muted"], "⏳", COLORS["bg_secondary"])

_WORKER_LABELS = {
    "scanning":  ("Scanning...",     COLORS["text_secondary"]),
    "unknown":   ("Unknown Person",  COLORS["accent_red"]),
    "spoof":     ("Spoof Detected",  COLORS["accent_red"]),
    "rfid":      ("Tap Card...",     COLORS["accent_orange"]),
    "rfid_fail": ("Card Not Found",  COLORS["accent_red"]),
    "liveness":  ("Verifying...",    COLORS["accent_purple"]),
}
_WORKER_DEFAULT = ("Waiting...", COLORS["text_muted"])


def eye_aspect_ratio(eye: np.ndarray) -> float:
    """Calculate the eye aspect ratio (EAR) for blink detection."""
    try:
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        C = dist.euclidean(eye[0], eye[3])
        if C < 1e-6:
            return 0.3
        return (A + B) / (2.0 * C)
    except Exception:
        return 0.3


# --- PRE-GENERATED VOICE COMMAND SYSTEM ---
# Uses pyttsx3 to generate WAV files at startup for zero-latency playback.

# Voice prompts for each state
_VOICE_PROMPTS: Dict[str, str] = {
    "granted":       "Access granted. Welcome.",
    "denied":        "Access denied.",
    "liveness":      "Please blink to verify.",
    "blink_ok":      "Blink detected.",
    "unknown":       "Face not recognized.",
    "spoof":         "Liveness check failed.",
    "rfid":          "Please tap your RFID card.",
    "rfid_fail":     "Card not registered.",
    "rfid_cooldown": "Card on cooldown. Please wait.",
    "rfid_auth_fail": "Card authentication failed. Possible clone detected.",
    "ppe_missing":   "Please put on required safety equipment.",
}

_VOICE_WAV_DIR: str = ""
_VOICE_WAV_PATHS: Dict[str, str] = {}


def _build_voice_cache() -> None:
    """Pre-generate all voice prompts as WAV files using pyttsx3."""
    global _VOICE_WAV_DIR
    _VOICE_WAV_DIR = tempfile.mkdtemp(prefix="smartgate_voice_")
    logger.info(f"Generating voice prompts in {_VOICE_WAV_DIR}...")

    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', VOICE_RATE)
        engine.setProperty('volume', VOICE_VOLUME)

        # Try to select a clear voice (prefer female/Zira on Windows)
        voices = engine.getProperty('voices')
        for voice in voices:
            if 'zira' in voice.name.lower() or 'female' in voice.name.lower():
                engine.setProperty('voice', voice.id)
                logger.info(f"Selected voice: {voice.name}")
                break
        else:
            if voices:
                engine.setProperty('voice', voices[0].id)
                logger.info(f"Using default voice: {voices[0].name}")

        for key, text in _VOICE_PROMPTS.items():
            wav_path = os.path.join(_VOICE_WAV_DIR, f"{key}.wav")
            engine.save_to_file(text, wav_path)
            _VOICE_WAV_PATHS[key] = wav_path

        engine.runAndWait()
        engine.stop()
        logger.info(f"Pre-generated {len(_VOICE_WAV_PATHS)} voice prompts")

    except Exception as e:
        logger.error(f"Voice generation failed: {e}. Audio will be disabled.")


_build_voice_cache()


def play_voice(prompt_key: str) -> None:
    """Play a pre-generated voice prompt asynchronously."""
    if not AUDIO_ENABLED:
        return
    wav_path = _VOICE_WAV_PATHS.get(prompt_key)
    if not wav_path or not os.path.exists(wav_path):
        logger.warning(f"Voice prompt not found: {prompt_key}")
        return

    def _play():
        try:
            winsound.PlaySound(wav_path, winsound.SND_FILENAME | winsound.SND_NODEFAULT)
        except Exception as e:
            logger.warning(f"Voice playback failed for '{prompt_key}': {e}")

    threading.Thread(target=_play, daemon=True).start()


class SmartGateClient(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()

        # Window setup
        self.title("Smart Gate - Face & PPE Verification")
        self.geometry("1000x700")
        self.minsize(800, 600)
        self.configure(fg_color=COLORS["bg_dark"])
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        # --- STATE MACHINE VARIABLES ---
        # States: IDLE, LIVENESS_CHECK, PPE_CHECK, GRANTED, UNKNOWN_FACE, SPOOF_DETECTED, RFID_FALLBACK, RFID_NOT_FOUND
        self.gate_state: str = "IDLE"
        self._prev_gate_state: str = ""  # Track state transitions to avoid redundant UI updates
        self.active_worker: Optional[str] = None
        self.state_timer: float = 0.0
        self.pending_encoding: Optional[np.ndarray] = None

        # Data State
        self.required_ids: List[int] = []
        self.known_encodings: List[np.ndarray] = []
        self.known_names: List[str] = []
        self.frame_count: int = 0
        self.last_results: List[Tuple[int, Tuple[int, int, int, int]]] = []
        self.last_face_locs: List[Tuple[int, int, int, int]] = []
        self.current_status: str = "waiting"

        # Liveness detection state
        self.blink_counter: int = 0
        self.blink_total: int = 0
        self.ear_below_thresh: bool = False
        self.liveness_prompted: bool = False
        self.baseline_ear: float = 0.3
        self.ear_samples: List[float] = []
        self.prev_gray_frame: Optional[np.ndarray] = None
        self.motion_detected: bool = False

        # Auto-brightness state
        self.current_exposure: float = 0.0
        self.brightness_frame_counter: int = 0

        # Resource management
        self.cap: Optional[cv2.VideoCapture] = None
        self.model: Optional[YOLO] = None
        self.executor: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
        self.model_loading: bool = True
        self.server_online: bool = True

        # HTTP session with connection pooling
        self._http_session: requests.Session = requests.Session()

        # UI resize debounce
        self._last_resize_time: float = 0.0
        self._cached_video_size: Tuple[int, int] = (640, 480)

        # Audio state (prevent repeated sounds)
        self._last_audio_state: str = ""

        # ESP32 state
        self._esp32_signal_sent: bool = False

        # RFID fallback state
        self._rfid_last_poll: float = 0.0
        self._rfid_polling: bool = False

        self._build_ui()
        self._initialize_resources()

    def _initialize_resources(self) -> None:
        """Initialize camera and AI model with error handling."""
        # Initialize camera
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                logger.error("Failed to open camera")
                self._set_status("denied", "CAMERA NOT AVAILABLE")
                self.cap = None
            else:
                self.current_exposure = self.cap.get(cv2.CAP_PROP_EXPOSURE)
                logger.info(f"Camera initialized with exposure: {self.current_exposure}")
        except Exception as e:
            logger.error(f"Camera initialization error: {e}")
            self._set_status("denied", "CAMERA ERROR")
            self.cap = None

        # Load model in background
        def load_model() -> None:
            try:
                self.model = YOLO(PPE_MODEL_PATH)
                self.model.to(DEVICE)
                self.model_loading = False
                logger.info(f"YOLO model loaded successfully on {DEVICE}")
            except Exception as e:
                logger.error(f"Failed to load YOLO model: {e}")
                self.model_loading = False
                self.after(0, lambda: self._set_status("denied", "AI MODEL LOAD FAILED"))

        self.executor.submit(load_model)

        # Single sync replaces separate sync_rules + sync_faces
        self._sync_server()
        self.update_loop()

    def _build_ui(self) -> None:
        self.main_container = ctk.CTkFrame(self, fg_color="transparent")
        self.main_container.pack(fill="both", expand=True, padx=20, pady=20)
        self._build_header()

        self.content = ctk.CTkFrame(self.main_container, fg_color="transparent")
        self.content.pack(fill="both", expand=True, pady=(15, 0))
        self.content.grid_columnconfigure(0, weight=3)
        self.content.grid_columnconfigure(1, weight=1)
        self.content.grid_rowconfigure(0, weight=1)

        self._build_video_section()
        self._build_sidebar()
        self._build_status_bar()

    def _build_header(self) -> None:
        """Header: Clean Slate 800 bar with Logo and Status Pill."""
        header = ctk.CTkFrame(self.main_container, fg_color="#1E293B", corner_radius=16, height=80, border_width=1, border_color="#334155")
        header.pack(fill="x", pady=(0, 20))
        # Use grid for 3-section layout
        header.pack_propagate(False)
        header.grid_columnconfigure(0, weight=1) # Left
        header.grid_columnconfigure(1, weight=0) # Spacer (Logo floats over)
        header.grid_columnconfigure(2, weight=1) # Right
        header.grid_rowconfigure(0, weight=1)

        # 1. Left: Brand Text
        brand_frame = ctk.CTkFrame(header, fg_color="transparent")
        brand_frame.grid(row=0, column=0, sticky="w", padx=25)
        ctk.CTkLabel(brand_frame, text="SITE COMMAND", font=("LunchType", 24, "bold"), text_color="#F8FAFC").pack(side="left")
        ctk.CTkLabel(brand_frame, text="  |  AI SURVEILLANCE", font=("Inter", 14), text_color="#94A3B8").pack(side="left", padx=(10, 0))

        # 2. Center: Logo (ABSOLUTE POSITIONING / ZERO-SHIFT)
        # We place this container relative to the header frame to ensure it is mathematically centered
        # regardless of the side content widths.
        logo_container = ctk.CTkFrame(header, fg_color="transparent")
        logo_container.place(relx=0.5, rely=0.5, anchor="center")
        
        logo_loaded = False
        try:
            logo_path = os.path.join("static", "images", "logo.jpeg")
            if os.path.exists(logo_path):
                pil_image = Image.open(logo_path).convert("RGB")
                
                # 1. Invert: Black Helmet -> White Helmet, White BG -> Black BG
                pil_image = ImageOps.invert(pil_image)
                
                # 2. Smart Transparency: Use the grayscale luminance as the Alpha channel
                # White pixels (helmet) become Opaque (255), Black pixels (bg) become Transparent (0)
                gray_mask = pil_image.convert("L")
                pil_image = pil_image.convert("RGBA")
                pil_image.putalpha(gray_mask)
                
                # 3. Resize to 90px (Dominant Anchor)
                size = (90, 90) 
                
                # High-Quality Lanczos Resampling
                output = ImageOps.fit(pil_image, size, method=Image.LANCZOS, centering=(0.5, 0.5))
                
                self.logo_image = ctk.CTkImage(light_image=output, dark_image=output, size=size)
                ctk.CTkLabel(logo_container, text="", image=self.logo_image).pack()
                logo_loaded = True
        except Exception as e:
            logger.warning(f"Logo load error: {e}")
        
        if not logo_loaded:
             ctk.CTkLabel(logo_container, text="🛡️", font=("LunchType", 40)).pack()

        # 3. Right: Server Status (or Action Buttons)
        status_frame = ctk.CTkFrame(header, fg_color="transparent")
        status_frame.grid(row=0, column=2, sticky="e", padx=25)

        self.server_indicator = ctk.CTkFrame(status_frame, fg_color="#0F172A", corner_radius=20, border_width=1, border_color="#334155")
        self.server_indicator.pack()
        
        self.server_status_dot = ctk.CTkLabel(self.server_indicator, text="●", font=("Inter", 12), text_color="#10B981")
        self.server_status_dot.pack(side="left", padx=(15, 5), pady=8)
        
        self.server_status_text = ctk.CTkLabel(self.server_indicator, text="SYSTEM ONLINE", font=("Inter", 11, "bold"), text_color="#CBD5E1")
        self.server_status_text.pack(side="left", padx=(0, 15), pady=8)

    def _build_video_section(self) -> None:
        """Video Section: Center stage, Slate 800 frame with rounded corners."""
        video_container = ctk.CTkFrame(self.content, fg_color="#1E293B", corner_radius=16, border_width=1, border_color="#334155")
        video_container.grid(row=0, column=0, sticky="nsew", padx=(0, 20))
        video_container.grid_rowconfigure(1, weight=1)
        video_container.grid_columnconfigure(0, weight=1)

        # Video Header
        video_header = ctk.CTkFrame(video_container, fg_color="transparent", height=50)
        video_header.grid(row=0, column=0, sticky="ew", padx=20, pady=(15, 5))
        ctk.CTkLabel(video_header, text="LIVE SENSOR FEED", font=("Inter", 13, "bold"), text_color="#94A3B8").pack(side="left")
        ctk.CTkLabel(video_header, text="CAM-01", font=("Inter", 11, "bold"), text_color="#64748B", fg_color="#0F172A", corner_radius=6).pack(side="right", ipadx=8, ipady=2)

        # Video Frame
        self.video_frame = ctk.CTkLabel(video_container, text="", fg_color="#000000", corner_radius=12)
        self.video_frame.grid(row=1, column=0, sticky="nsew", padx=20, pady=(0, 20))

    def _build_sidebar(self) -> None:
        """Sidebar: Stacked Cards architecture with spacing."""
        sidebar = ctk.CTkFrame(self.content, fg_color="transparent", width=320)
        sidebar.grid(row=0, column=1, sticky="nsew")
        sidebar.grid_propagate(False)

        # 1. Identity Card
        id_card = ctk.CTkFrame(sidebar, fg_color="#1E293B", corner_radius=16, border_width=1, border_color="#334155")
        id_card.pack(fill="x", pady=(0, 15))
        
        ctk.CTkLabel(id_card, text="PERSONNEL IDENTITY", font=("Inter", 11, "bold"), text_color="#64748B").pack(anchor="w", padx=20, pady=(20, 5))
        self.worker_name_label = ctk.CTkLabel(id_card, text="SCANNING...", font=("LunchType", 22, "bold"), text_color="#F8FAFC", wraplength=280)
        self.worker_name_label.pack(anchor="w", padx=20, pady=(0, 20))

        # 2. Liveness & PPE Card (Combined for density)
        status_card = ctk.CTkFrame(sidebar, fg_color="#1E293B", corner_radius=16, border_width=1, border_color="#334155")
        status_card.pack(fill="x", pady=(0, 15))

        # Liveness Row
        ctk.CTkLabel(status_card, text="LIVENESS CHECK", font=("Inter", 11, "bold"), text_color="#64748B").pack(anchor="w", padx=20, pady=(20, 5))
        self.liveness_status_label = ctk.CTkLabel(status_card, text="--", font=("Inter", 14), text_color="#CBD5E1")
        self.liveness_status_label.pack(anchor="w", padx=20, pady=(0, 15))
        
        # Divider
        ctk.CTkFrame(status_card, height=1, fg_color="#334155").pack(fill="x", padx=20)

        # PPE Row
        ctk.CTkLabel(status_card, text="PPE COMPLIANCE", font=("Inter", 11, "bold"), text_color="#64748B").pack(anchor="w", padx=20, pady=(15, 5))
        self.ppe_status_frame = ctk.CTkFrame(status_card, fg_color="transparent")
        self.ppe_status_frame.pack(fill="x", padx=20, pady=(0, 20))
        self.ppe_status_label = ctk.CTkLabel(self.ppe_status_frame, text="--", font=("Inter", 14), text_color="#CBD5E1")
        self.ppe_status_label.pack(anchor="w")

        # 3. Requirements Card
        req_card = ctk.CTkFrame(sidebar, fg_color="#1E293B", corner_radius=16, border_width=1, border_color="#334155")
        req_card.pack(fill="both", expand=True, pady=(0, 15))
        
        ctk.CTkLabel(req_card, text="REQUIRED PROTOCOLS", font=("Inter", 11, "bold"), text_color="#64748B").pack(anchor="w", padx=20, pady=(20, 10))
        self.required_ppe_frame = ctk.CTkFrame(req_card, fg_color="transparent")
        self.required_ppe_frame.pack(fill="both", expand=True, padx=20, pady=(0, 20))

        # 4. Timer (Bottom pinned)
        self.time_label = ctk.CTkLabel(sidebar, text="00:00:00", font=("LunchType", 32, "bold"), text_color="#475569")
        self.time_label.pack(side="bottom", anchor="e", pady=10)
        self._update_time()

    def _build_status_bar(self) -> None:
        """Status Bar: High-visibility bottom notification area."""
        self.status_bar = ctk.CTkFrame(self.main_container, fg_color="#1E293B", corner_radius=16, height=70, border_width=1, border_color="#334155")
        self.status_bar.pack(fill="x", pady=(20, 0))
        self.status_bar.pack_propagate(False)

        container = ctk.CTkFrame(self.status_bar, fg_color="transparent")
        container.pack(expand=True, fill="both", padx=25)

        self.status_icon = ctk.CTkLabel(container, text="🔍", font=("Inter", 28))
        self.status_icon.pack(side="left")

        self.status_text = ctk.CTkLabel(container, text="SYSTEM READY", font=("LunchType", 18, "bold"), text_color="#F8FAFC")
        self.status_text.pack(side="left", padx=(20, 0))

        # Action hint
        ctk.CTkLabel(container, text="FACE CAMERA FOR SCAN", font=("Inter", 12, "bold"), text_color="#64748B").pack(side="right")

    def _update_time(self) -> None:
        self.time_label.configure(text=datetime.now().strftime("%H:%M:%S"))
        self.after(1000, self._update_time)

    def _update_required_ppe_display(self) -> None:
        for widget in self.required_ppe_frame.winfo_children():
            widget.destroy()
        if not self.required_ids:
            ctk.CTkLabel(self.required_ppe_frame, text="No requirements", font=ctk.CTkFont(size=12),
                         text_color=COLORS["text_muted"]).pack(anchor="w")
        else:
            for ppe_id in self.required_ids[:5]:
                name = PPE_NAMES.get(ppe_id, f"ID {ppe_id}")
                badge = ctk.CTkFrame(self.required_ppe_frame, fg_color=COLORS["bg_secondary"], corner_radius=6)
                badge.pack(anchor="w", pady=2)
                ctk.CTkLabel(badge, text=f"  {name}  ", font=ctk.CTkFont(size=11),
                             text_color=COLORS["text_secondary"]).pack(padx=8, pady=3)

    def _update_server_status(self, online: bool) -> None:
        """Update server connection status indicator."""
        if self.server_online == online:
            return  # Skip redundant UI update
        self.server_online = online
        if online:
            self.server_status_dot.configure(text_color=COLORS["accent_green"])
            self.server_status_text.configure(text="Online")
        else:
            self.server_status_dot.configure(text_color=COLORS["accent_red"])
            self.server_status_text.configure(text="Offline")

    def _play_audio_for_state(self, state: str) -> None:
        """Play voice prompt for state transitions (once per state)."""
        if state == self._last_audio_state:
            return
        self._last_audio_state = state

        # Map states to voice prompt keys
        _STATE_TO_VOICE = {
            "granted":       "granted",
            "denied":        "denied",
            "unknown":       "unknown",
            "spoof":         "spoof",
            "rfid_fail":     "rfid_fail",
            "rfid_cooldown": "rfid_cooldown",
            "rfid_auth_fail": "rfid_auth_fail",
            "rfid":          "rfid",
            "liveness":      "liveness",
        }
        voice_key = _STATE_TO_VOICE.get(state)
        if voice_key:
            play_voice(voice_key)

    def _poll_esp32_rfid(self) -> Optional[Tuple[str, str]]:
        """Poll the ESP32 for a scanned RFID card. Returns (UID, token) tuple or None."""
        now = time.time()
        if now - self._rfid_last_poll < (RFID_POLL_INTERVAL_MS / 1000.0):
            return None
        self._rfid_last_poll = now

        try:
            response = self._http_session.get(f"http://{ESP32_IP}:{ESP32_PORT}/rfid", timeout=1)
            if response.ok:
                data = response.json()
                uid = data.get('uid', '')
                token = data.get('token', '')
                if uid:
                    logger.info(f"RFID card detected: {uid} (token: {'yes' if token else 'none'})")
                    return (uid, token)
        except requests.ConnectionError:
            logger.debug(f"ESP32 RFID poll: cannot reach {ESP32_IP}")
        except requests.Timeout:
            logger.debug("ESP32 RFID poll: timeout")
        except Exception as e:
            logger.warning(f"RFID poll error: {e}")
        return None

    def _lookup_rfid_card(self, uid: str, token: str = "") -> Optional[str]:
        """Look up an RFID card UID + token on the server. Returns worker name, 'COOLDOWN', 'AUTH_FAIL', or None."""
        try:
            response = self._http_session.get(
                f"{SERVER_URL}/api/rfid_lookup",
                params={'uid': uid, 'token': token},
                timeout=API_TIMEOUT_SECONDS
            )
            if response.ok:
                data = response.json()
                if data.get('found'):
                    if data.get('auth_failed'):
                        logger.warning(f"RFID auth failed for {uid}: {data.get('error', '')}")
                        return "AUTH_FAIL"
                    if data.get('cooldown'):
                        remaining = data.get('remaining_minutes', '?')
                        logger.warning(f"RFID card {uid} on cooldown ({remaining} min remaining)")
                        return "COOLDOWN"
                    return data.get('name')
        except Exception as e:
            logger.warning(f"RFID lookup error: {e}")
        return None

    def _set_status(self, status: str, message: str, worker_name: str = "") -> None:
        self.current_status = status

        color, icon, bg = _STATUS_VISUALS.get(status, _STATUS_DEFAULT)
        self.status_icon.configure(text=icon)
        self.status_text.configure(text=message, text_color=color)
        self.status_bar.configure(fg_color=bg)

        if worker_name:
            self.worker_name_label.configure(text=worker_name, text_color=COLORS["text_primary"])
        else:
            text, clr = _WORKER_LABELS.get(status, _WORKER_DEFAULT)
            self.worker_name_label.configure(text=text, text_color=clr)

    def _log_access(self, name: str, status: str, details: str) -> None:
        """Log an access event to the server asynchronously."""
        def _send_log():
            try:
                self._http_session.post(
                    f"{SERVER_URL}/api/log",
                    json={"name": name, "status": status, "details": details},
                    timeout=API_TIMEOUT_SECONDS
                )
            except Exception as e:
                logger.warning(f"Failed to log access event: {e}")

        self.executor.submit(_send_log)

    def _send_esp32_signal(self) -> None:
        """Send '1' to ESP32 over WiFi to open the gate."""
        if self._esp32_signal_sent:
            return
        self._esp32_signal_sent = True

        def _send():
            try:
                response = self._http_session.post(ESP32_URL, data="1", timeout=3)
                if response.ok:
                    logger.info(f"ESP32 gate signal sent successfully to {ESP32_IP}")
                else:
                    logger.warning(f"ESP32 responded with status {response.status_code}")
            except requests.ConnectionError:
                logger.warning(f"Cannot reach ESP32 at {ESP32_IP} - check WiFi connection")
            except requests.Timeout:
                logger.warning(f"ESP32 at {ESP32_IP} timed out")
            except Exception as e:
                logger.warning(f"ESP32 signal failed: {e}")

        self.executor.submit(_send)

    def _adjust_brightness(self, frame: np.ndarray) -> None:
        """Analyze frame brightness and adjust camera exposure if needed."""
        self.brightness_frame_counter += 1
        if self.brightness_frame_counter % BRIGHTNESS_CHECK_INTERVAL != 0:
            return

        if self.cap is None:
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        avg_brightness = np.mean(gray)

        try:
            if avg_brightness < TARGET_BRIGHTNESS_MIN:
                new_exposure = self.current_exposure + EXPOSURE_STEP
                self.cap.set(cv2.CAP_PROP_EXPOSURE, new_exposure)
                self.current_exposure = new_exposure
                logger.debug(f"Brightness {avg_brightness:.1f} too low, increasing exposure to {new_exposure}")
            elif avg_brightness > TARGET_BRIGHTNESS_MAX:
                new_exposure = self.current_exposure - EXPOSURE_STEP
                self.cap.set(cv2.CAP_PROP_EXPOSURE, new_exposure)
                self.current_exposure = new_exposure
                logger.debug(f"Brightness {avg_brightness:.1f} too high, decreasing exposure to {new_exposure}")
        except Exception as e:
            logger.warning(f"Failed to adjust exposure: {e}")

    def _detect_blink(self, rgb_frame: np.ndarray, face_landmarks: List) -> bool:
        """Detect if a blink occurred using EAR with adaptive threshold + delta detection."""
        if not face_landmarks:
            return False

        try:
            landmarks = face_landmarks[0]

            left_eye = np.array(landmarks['left_eye'])
            right_eye = np.array(landmarks['right_eye'])

            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)
            ear = (left_ear + right_ear) / 2.0

            # Calibrate baseline EAR from first few samples
            if len(self.ear_samples) < 5:
                self.ear_samples.append(ear)
                if len(self.ear_samples) == 5:
                    self.baseline_ear = np.mean(self.ear_samples)
                    logger.info(f"EAR baseline calibrated: {self.baseline_ear:.3f}")
                return False

            self.ear_samples.append(ear)
            if len(self.ear_samples) > 30:
                self.ear_samples = self.ear_samples[-30:]
            running_avg = np.mean(self.ear_samples[-15:])

            adaptive_thresh = self.baseline_ear * 0.80
            threshold = max(adaptive_thresh, EYE_AR_THRESH)

            ear_drop = ear < (running_avg * EAR_DROP_RATIO)

            if ear < threshold or ear_drop:
                self.blink_counter += 1
                if ear_drop:
                    logger.debug(f"EAR delta drop: {ear:.3f} vs avg {running_avg:.3f}")
                else:
                    logger.debug(f"EAR below threshold: {ear:.3f} < {threshold:.3f}")
            else:
                if self.blink_counter >= EYE_AR_CONSEC_FRAMES:
                    self.blink_total += 1
                    logger.info(f"Blink detected! EAR: {ear:.3f}, threshold: {threshold:.3f}, Total: {self.blink_total}")
                    play_voice("blink_ok")
                self.blink_counter = 0

            return self.blink_total >= REQUIRED_BLINKS
        except Exception as e:
            logger.warning(f"Blink detection error: {e}")
            return False

    def _detect_motion(self, gray_frame: np.ndarray) -> bool:
        """Detect significant motion as alternative liveness check."""
        if self.prev_gray_frame is None:
            self.prev_gray_frame = gray_frame.copy()
            return False

        try:
            frame_diff = cv2.absdiff(self.prev_gray_frame, gray_frame)
            _, thresh = cv2.threshold(frame_diff, MOTION_THRESHOLD, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                if cv2.contourArea(contour) > MOTION_AREA_THRESHOLD:
                    logger.info("Motion detected - liveness confirmed")
                    self.prev_gray_frame = gray_frame.copy()
                    return True

            self.prev_gray_frame = gray_frame.copy()
            return False
        except Exception as e:
            logger.warning(f"Motion detection error: {e}")
            return False

    def _reset_liveness_state(self) -> None:
        """Reset liveness detection state."""
        self.blink_counter = 0
        self.blink_total = 0
        self.ear_below_thresh = False
        self.liveness_prompted = False
        self.pending_encoding = None
        self.baseline_ear = 0.3
        self.ear_samples = []
        self.prev_gray_frame = None
        self.motion_detected = False

    def _sync_server(self) -> None:
        """Unified server sync: fetch rules + faces in a single API call."""
        def _fetch() -> None:
            try:
                r = self._http_session.get(f"{SERVER_URL}/api/sync", timeout=API_TIMEOUT_SECONDS)
                if r.ok:
                    data = r.json()
                    # Update rules
                    self.required_ids = data['required_ppe']
                    self.after(0, self._update_required_ppe_display)
                    # Update faces
                    self.known_names = [u['name'] for u in data['faces']]
                    self.known_encodings = [np.array(u['encoding']) for u in data['faces']]
                    self.after(0, lambda: self._update_server_status(True))
            except requests.RequestException as e:
                logger.warning(f"Failed to sync with server: {e}")
                self.after(0, lambda: self._update_server_status(False))

        self.executor.submit(_fetch)
        self.after(RULES_SYNC_INTERVAL_MS, self._sync_server)

    def _get_debounced_video_size(self) -> Tuple[int, int]:
        """Get video size with debouncing to avoid excessive resizing."""
        current_time = time.time()
        if current_time - self._last_resize_time > 0.1:
            video_width = self.video_frame.winfo_width()
            video_height = self.video_frame.winfo_height()
            if video_width > 100 and video_height > 100:
                self._cached_video_size = (video_width, video_height)
                self._last_resize_time = current_time
        return self._cached_video_size

    def update_loop(self) -> None:
        # Check if camera and model are ready
        if self.cap is None or not self.cap.isOpened():
            self.after(UPDATE_LOOP_DELAY_MS, self.update_loop)
            return

        if self.model_loading:
            self.video_frame.configure(text="Loading AI Model...")
            self.after(UPDATE_LOOP_DELAY_MS, self.update_loop)
            return

        ret, frame = self.cap.read()
        if not ret:
            logger.warning("Failed to read frame from camera")
            self.after(UPDATE_LOOP_DELAY_MS, self.update_loop)
            return

        self.frame_count += 1

        # Auto-brightness adjustment
        self._adjust_brightness(frame)

        # Prepare frame for AI processing
        small_frame = cv2.resize(frame, AI_RESOLUTION)
        scale_x = frame.shape[1] / AI_RESOLUTION[0]
        scale_y = frame.shape[0] / AI_RESOLUTION[1]
        rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Run non-liveness AI states only periodically for performance
        run_ai_this_frame = (self.frame_count % PROCESS_EVERY_N_FRAMES == 0)

        # ==========================================
        # STATE: LIVENESS CHECK (Blink Detection)
        # Runs on EVERY frame to avoid missing blinks
        # ==========================================
        if self.gate_state == "LIVENESS_CHECK":
            remaining = LIVENESS_TIMEOUT_SEC - (time.time() - self.state_timer)
            self._set_status("liveness", f"PLEASE BLINK ({remaining:.0f}s)", self.active_worker)
            self.liveness_status_label.configure(
                text=f"Blinks: {self.blink_total}/{REQUIRED_BLINKS}",
                text_color=COLORS["accent_purple"]
            )

            # Use higher resolution for reliable landmark detection
            liveness_frame = cv2.resize(frame, LIVENESS_RESOLUTION)
            liveness_rgb = cv2.cvtColor(liveness_frame, cv2.COLOR_BGR2RGB)
            liveness_scale_x = frame.shape[1] / LIVENESS_RESOLUTION[0]
            liveness_scale_y = frame.shape[0] / LIVENESS_RESOLUTION[1]

            # Get face landmarks for blink detection at higher resolution
            locs = face_recognition.face_locations(liveness_rgb, model='hog')
            if locs:
                landmarks = face_recognition.face_landmarks(liveness_rgb, locs)

                # Update face location for drawing (scale from liveness resolution)
                self.last_face_locs = [(int(t * liveness_scale_y), int(r * liveness_scale_x),
                                        int(b * liveness_scale_y), int(l * liveness_scale_x))
                                       for (t, r, b, l) in locs]

                # Check for blink using high-res landmarks
                if self._detect_blink(liveness_rgb, landmarks):
                    # Liveness verified!
                    self.gate_state = "PPE_CHECK"
                    self.liveness_status_label.configure(text="Verified ✓", text_color=COLORS["accent_green"])
                    self.last_face_locs = []

            # Timeout - spoof detected
            if time.time() - self.state_timer > LIVENESS_TIMEOUT_SEC:
                self.gate_state = "SPOOF_DETECTED"
                self.state_timer = time.time()
                self._play_audio_for_state("spoof")

        # All other states run periodically for performance
        elif run_ai_this_frame:

            # ==========================================
            # STATE: IDLE / SCANNING FACE
            # ==========================================
            if self.gate_state == "IDLE":
                # Only reset UI labels on state transition
                if self._prev_gate_state != "IDLE":
                    self._set_status("scanning", "PLEASE SCAN FACE")
                    self.ppe_status_label.configure(text="--", text_color=COLORS["text_muted"])
                    self.liveness_status_label.configure(text="--", text_color=COLORS["text_muted"])
                    self._reset_liveness_state()

                locs = face_recognition.face_locations(rgb)
                encs = face_recognition.face_encodings(rgb, locs)

                self.last_face_locs = [(int(t * scale_y), int(r * scale_x), int(b * scale_y), int(l * scale_x)) for
                                       (t, r, b, l) in locs]
                self.last_results = []

                if encs and self.known_encodings:
                    for enc in encs:
                        # Use vectorized face_distance instead of compare_faces loop
                        distances = face_recognition.face_distance(self.known_encodings, enc)
                        best_idx = np.argmin(distances)
                        if distances[best_idx] <= FACE_DISTANCE_TOLERANCE:
                            self.active_worker = self.known_names[best_idx]
                            self.pending_encoding = enc
                            self.gate_state = "LIVENESS_CHECK"
                            self.state_timer = time.time()
                            self._play_audio_for_state("liveness")
                            break
                    else:
                        self.gate_state = "UNKNOWN_FACE"
                        self.state_timer = time.time()
                        self._play_audio_for_state("unknown")
                elif encs:
                    # Known encodings empty
                    self.gate_state = "UNKNOWN_FACE"
                    self.state_timer = time.time()
                    self._play_audio_for_state("unknown")

            # ==========================================
            # STATE: SPOOF DETECTED
            # ==========================================
            elif self.gate_state == "SPOOF_DETECTED":
                self._set_status("spoof", "LIVENESS CHECK FAILED - TAP RFID")
                self.liveness_status_label.configure(text="Failed ✗", text_color=COLORS["accent_red"])
                self.ppe_status_label.configure(text="Use RFID fallback", text_color=COLORS["accent_orange"])

                cv2.putText(frame, "SPOOF DETECTED", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

                if time.time() - self.state_timer > SPOOF_DISPLAY_SEC:
                    self.gate_state = "RFID_FALLBACK"
                    self.state_timer = time.time()
                    self._rfid_last_poll = 0.0
                    self._reset_liveness_state()
                    self._play_audio_for_state("rfid")
                    self._log_access(self.active_worker or "Unknown", "DENIED", "Liveness check failed (spoof)")

            # ==========================================
            # STATE: UNKNOWN FACE DETECTED
            # ==========================================
            elif self.gate_state == "UNKNOWN_FACE":
                self._set_status("unknown", "FACE NOT RECOGNIZED")
                self.ppe_status_label.configure(text="Use RFID fallback", text_color=COLORS["accent_orange"])
                self.liveness_status_label.configure(text="--", text_color=COLORS["text_muted"])

                for (t, r, b, l) in self.last_face_locs:
                    cv2.rectangle(frame, (l, t), (r, b), (0, 0, 255), 2)
                    cv2.putText(frame, "UNKNOWN", (l, t - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                if time.time() - self.state_timer > UNKNOWN_FACE_TIMEOUT_SEC:
                    self.gate_state = "RFID_FALLBACK"
                    self.state_timer = time.time()
                    self._rfid_last_poll = 0.0
                    self.last_face_locs = []
                    self._play_audio_for_state("rfid")
                    self._log_access("Unknown", "DENIED", "Face not recognized")

            # ==========================================
            # STATE: RFID FALLBACK (Waiting for card tap)
            # ==========================================
            elif self.gate_state == "RFID_FALLBACK":
                remaining = RFID_TIMEOUT_SEC - (time.time() - self.state_timer)
                self._set_status("rfid", f"TAP YOUR RFID CARD ({remaining:.0f}s)")
                self.liveness_status_label.configure(text="RFID Mode", text_color=COLORS["accent_orange"])
                self.ppe_status_label.configure(text="Waiting for card...", text_color=COLORS["accent_orange"])

                cv2.putText(frame, "TAP RFID CARD", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 3)

                rfid_result = self._poll_esp32_rfid()
                if rfid_result:
                    uid, token = rfid_result
                    worker_name = self._lookup_rfid_card(uid, token)
                    if worker_name == "AUTH_FAIL":
                        logger.warning(f"RFID card {uid} failed sector key auth (possible clone)")
                        self.gate_state = "RFID_NOT_FOUND"
                        self.state_timer = time.time()
                        self._last_audio_state = ""  # Reset so voice plays
                        self._play_audio_for_state("rfid_auth_fail")
                        self._log_access("Unknown", "DENIED", f"RFID auth failed (clone?): {uid}")
                    elif worker_name == "COOLDOWN":
                        logger.warning(f"RFID card {uid} is on cooldown")
                        self.gate_state = "RFID_NOT_FOUND"
                        self.state_timer = time.time()
                        self._last_audio_state = ""  # Reset so voice plays
                        self._play_audio_for_state("rfid_cooldown")
                        self._log_access("Unknown", "DENIED", f"RFID card {uid} on cooldown")
                    elif worker_name:
                        logger.info(f"RFID verified: {worker_name} (UID: {uid})")
                        self.active_worker = worker_name
                        self.gate_state = "PPE_CHECK"
                        self.liveness_status_label.configure(
                            text=f"RFID: {uid}", text_color=COLORS["accent_green"]
                        )
                        self._log_access(worker_name, "RFID_VERIFIED", f"Card UID: {uid}")
                    else:
                        logger.warning(f"RFID card {uid} not registered")
                        self.gate_state = "RFID_NOT_FOUND"
                        self.state_timer = time.time()
                        self._play_audio_for_state("denied")
                        self._log_access("Unknown", "DENIED", f"Unregistered RFID card: {uid}")

                if remaining <= 0:
                    self.gate_state = "IDLE"
                    self._last_audio_state = ""

            # ==========================================
            # STATE: RFID CARD NOT REGISTERED
            # ==========================================
            elif self.gate_state == "RFID_NOT_FOUND":
                self._set_status("rfid_fail", "CARD NOT REGISTERED")
                self.liveness_status_label.configure(text="--", text_color=COLORS["text_muted"])
                self.ppe_status_label.configure(text="Access Denied", text_color=COLORS["accent_red"])

                cv2.putText(frame, "CARD NOT REGISTERED", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                if time.time() - self.state_timer > RFID_CARD_NOT_FOUND_DISPLAY_SEC:
                    self.gate_state = "IDLE"
                    self._last_audio_state = ""

            # ==========================================
            # STATE: CHECKING PPE (Worker Verified)
            # ==========================================
            elif self.gate_state == "PPE_CHECK":
                if self.model is None:
                    self.gate_state = "IDLE"
                    self.after(UPDATE_LOOP_DELAY_MS, self.update_loop)
                    return

                results = self.model(small_frame, stream=True, verbose=False, device=DEVICE)
                current_detected: Set[int] = set()
                self.last_results = []

                for r in results:
                    for box in r.boxes:
                        if float(box.conf[0]) > CONFIDENCE:
                            cls = int(box.cls[0])
                            current_detected.add(cls)
                            x1, y1, x2, y2 = box.xyxy[0]
                            self.last_results.append(
                                (cls, (int(x1 * scale_x), int(y1 * scale_y), int(x2 * scale_x), int(y2 * scale_y))))

                missing = [PPE_NAMES.get(id, str(id)) for id in self.required_ids if id not in current_detected]

                if missing:
                    self._set_status("warning", f"PUT ON: {', '.join(missing[:2])}", self.active_worker)
                    self.ppe_status_label.configure(text=f"Missing {len(missing)} items",
                                                    text_color=COLORS["accent_orange"])
                else:
                    self.gate_state = "GRANTED"
                    self.state_timer = time.time()
                    self._play_audio_for_state("granted")
                    self._send_esp32_signal()
                    self._log_access(self.active_worker or "Unknown", "GRANTED", "Face + PPE verified")

            # ==========================================
            # STATE: ACCESS GRANTED (Timer)
            # ==========================================
            elif self.gate_state == "GRANTED":
                self._set_status("granted", "ACCESS GRANTED", self.active_worker)
                self.ppe_status_label.configure(text="All Checks Passed", text_color=COLORS["accent_green"])

                if time.time() - self.state_timer > GRANT_DISPLAY_DURATION_SEC:
                    self.gate_state = "IDLE"
                    self.active_worker = None
                    self.last_results = []
                    self._last_audio_state = ""
                    self._esp32_signal_sent = False

        # Track state transitions for IDLE optimization
        self._prev_gate_state = self.gate_state

        # --- DRAWING ---
        for cls, (x1, y1, x2, y2) in self.last_results:
            label = PPE_NAMES.get(cls, "")
            color = (16, 185, 129) if self.gate_state == "GRANTED" else (0, 255, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Draw face box based on state
        if self.gate_state == "IDLE":
            for (t, r, b, l) in self.last_face_locs:
                cv2.rectangle(frame, (l, t), (r, b), (255, 255, 255), 2)
        elif self.gate_state == "LIVENESS_CHECK":
            for (t, r, b, l) in self.last_face_locs:
                cv2.rectangle(frame, (l, t), (r, b), (139, 92, 246), 2)
                cv2.putText(frame, "BLINK NOW", (l, t - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (139, 92, 246), 2)

        # Display Image using debounced video size
        display_size = self._get_debounced_video_size()
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        ctk_img = ctk.CTkImage(img, size=display_size)
        self.video_frame.configure(image=ctk_img, text="")
        self.video_frame.image = ctk_img

        self.after(UPDATE_LOOP_DELAY_MS, self.update_loop)

    def destroy(self) -> None:
        """Clean up resources before destroying the window."""
        logger.info("Cleaning up resources...")

        if self.cap is not None:
            try:
                self.cap.release()
                logger.info("Camera released")
            except Exception as e:
                logger.error(f"Error releasing camera: {e}")

        try:
            self.executor.shutdown(wait=False)
            logger.info("Thread pool shut down")
        except Exception as e:
            logger.error(f"Error shutting down thread pool: {e}")

        try:
            self._http_session.close()
        except Exception:
            pass

        super().destroy()


if __name__ == "__main__":
    app = SmartGateClient()
    app.mainloop()