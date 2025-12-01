import sys, os, cv2, time, threading, logging, re, collections
import numpy as np
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QEasingCurve, pyqtProperty, QPropertyAnimation, QRectF
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QComboBox, QPushButton, QTextEdit, QSlider, QGroupBox, QSizePolicy,
    QSpacerItem, QCheckBox, QDoubleSpinBox, QGridLayout, QMessageBox, QFileDialog,
    QFrame
)
from PyQt5.QtGui import (
    QPalette, QColor, QImage, QPixmap, QBrush, QLinearGradient, QPen, QPainter,
    QIcon, QFont, QPolygon, QCursor
)

import mediapipe as mp
import mido
try:
    from groq import Groq
except Exception:
    Groq = None
logging.basicConfig(level=logging.INFO, format='%(asctime)s — %(levelname)s — %(message)s')
script_dir = os.path.dirname(os.path.realpath(__file__))
# ---- UI constants (keeps layout predictable) ----
VIDEO_W, VIDEO_H = 640, 360     # camera preview size (16:9)
LEFT_COLUMN_MAX_W = 460         # compact control panel
NOTE_GUIDE_W = 120              # slim note guide
# ----------------------------- Utilities ---------------------------------
def midi_note_to_name(n: int) -> str:
    if n is None or n <= 0:
        return "--"
    names = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
    return f"{names[n%12]}{(n//12)-1}"
# -------------------------- Recording Manager -----------------------------
class RecordingManager:
    """
    Thread-safe collector of timestamped MIDI events; exports a standard .mid file.
    NOTE: never call parent.update_log() while holding the lock.
    """
    def __init__(self, parent=None):
        self.lock = threading.Lock()
        self.parent = parent
        self.reset()
    def reset(self):
        with self.lock:
            self.recording = False
            self.start_time = None
            self.events = []   # dicts: {time, type('on'|'off'), note, velocity, channel}
            self.saved_path = None
    def start_recording(self):
        with self.lock:
            self.recording = False
            self.start_time = None
            self.events = []
            self.saved_path = None
            self.recording = True
            self.start_time = time.time()
        if self.parent:
            self.parent.update_log("Recording started...")
    def stop_recording(self):
        with self.lock:
            active = self.recording
            n = len(self.events)
            self.recording = False
        if self.parent:
            self.parent.update_log("Recording stopped. Events captured: {}".format(n if active else 0))
    def is_recording(self):
        with self.lock:
            return self.recording
    def has_events(self):
        with self.lock:
            return len(self.events) > 0
    def log_event(self, note, velocity, event_type='on', channel=0, timestamp=None):
        if timestamp is None:
            timestamp = time.time()
        with self.lock:
            if not self.recording:
                return
            rel = timestamp - self.start_time if self.start_time else 0.0
            self.events.append({
                "time": float(rel),
                "type": 'on' if event_type == 'on' else 'off',
                "note": int(note),
                "velocity": int(velocity),
                "channel": int(channel)
            })
    def save_to_midi(self, parent_widget=None):
        with self.lock:
            if not self.events:
                return None
            events = sorted(self.events, key=lambda e: e['time'])
        mid = mido.MidiFile(ticks_per_beat=480)
        track = mido.MidiTrack()
        mid.tracks.append(track)
        def seconds_to_ticks(seconds, tempo_us=500000, tpq=480):
            return int(round(seconds * (tpq * 1e6) / tempo_us))
        last_time = 0.0
        for ev in events:
            delta_sec = ev['time'] - last_time
            last_time = ev['time']
            delta_ticks = seconds_to_ticks(delta_sec)
            if ev['type'] == 'on':
                msg = mido.Message('note_on', note=ev['note'], velocity=ev['velocity'],
                                   channel=ev['channel'], time=delta_ticks)
            else:
                msg = mido.Message('note_off', note=ev['note'], velocity=ev['velocity'],
                                   channel=ev['channel'], time=delta_ticks)
            track.append(msg)
        track.append(mido.MetaMessage('end_of_track', time=0))
        suggested = time.strftime("gesture_recording_%Y%m%d_%H%M%S.mid", time.localtime())
        save_path, _ = QFileDialog.getSaveFileName(parent_widget, "Save Recording", suggested, "MIDI files (*.mid)")
        if save_path:
            try:
                mid.save(save_path)
                with self.lock:
                    self.saved_path = save_path
                if self.parent:
                    self.parent.update_log(f"Recording saved to: {save_path}")
                return save_path
            except Exception as e:
                if self.parent:
                    self.parent.update_log(f"Failed to save MIDI: {e}")
        return None
# --------------------------- Musical Feedback Manager --------------------------------
class MusicalFeedbackManager:
    """Provides musical insights after performance sessions rather than parameter tweaks."""
    def __init__(self, parent, video_thread):
        self.parent = parent
        self.video_thread = video_thread
        self.client = None
        self.is_active = False
        self.gesture_log = []
        self.key = os.getenv("GROQ_API_KEY")
        
        if self.key and Groq is not None:
            try:
                self.client = Groq(api_key=self.key)
                if self.parent:
                    self.parent.update_log("LLM Musical Feedback ready (disabled by default).")
            except Exception as e:
                if self.parent:
                    self.parent.update_log(f"LLM Musical Feedback: could not init Groq: {e}")
        else:
            if self.parent:
                self.parent.update_log("LLM Musical Feedback ready (disabled by default).")

    def add_gesture_data(self, note, velocity, timestamp=None):
        """Collect data during performance if feedback mode is on."""
        if not self.is_active:
            return
        if timestamp is None:
            timestamp = time.time()
        self.gesture_log.append({
            "timestamp": timestamp,
            "note": note,
            "velocity": velocity
        })
        # Keep log size manageable
        if len(self.gesture_log) > 2000:
            self.gesture_log = self.gesture_log[-2000:]

    def analyze_session(self):
        """Analyze full session after STOP and provide musical feedback."""
        if not self.is_active or not self.client or not self.gesture_log:
            return
            
        if len(self.gesture_log) < 30:  # Need enough data for meaningful analysis
            self.parent.update_log("LLM Feedback: insufficient gesture data for analysis.")
            return
            
        if self.parent:
            self.parent.update_log("LLM Musical Feedback: analyzing your performance…")
            
        try:
            logs = self.gesture_log
            # Calculate meaningful musical metrics
            notes = [x["note"] for x in logs]
            velocities = [x["velocity"] for x in logs]
            timestamps = [x["timestamp"] for x in logs]
            
            # Note range analysis
            note_range = max(notes) - min(notes)
            avg_note = sum(notes) / len(notes)
            min_note_name = midi_note_to_name(min(notes))
            max_note_name = midi_note_to_name(max(notes))
            
            # Velocity analysis
            avg_velocity = sum(velocities) / len(velocities)
            vel_range = max(velocities) - min(velocities)
            vel_std = np.std(velocities) if len(velocities) > 1 else 0
            vel_stability = max(0, 100 - (vel_std * 1.2))  # Normalize to percentage
            
            # Movement patterns
            upward_moves = 0
            downward_moves = 0
            for i in range(1, len(notes)):
                if notes[i] > notes[i-1] + 3:  # Significant upward movement
                    upward_moves += 1
                elif notes[i] < notes[i-1] - 3:  # Significant downward movement
                    downward_moves += 1
            
            movement_ratio = upward_moves / (downward_moves + 1)  # Avoid division by zero
            
            # Tempo estimation (based on significant note changes)
            significant_changes = []
            last_note = notes[0]
            for i in range(1, len(notes)):
                if abs(notes[i] - last_note) > 5:  # Significant change
                    time_diff = timestamps[i] - timestamps[i-1]
                    if 0.1 < time_diff < 2.0:  # Reasonable time between notes
                        significant_changes.append(1/time_diff)  # Changes per second
                    last_note = notes[i]
            
            avg_changes_per_sec = sum(significant_changes) / len(significant_changes) if significant_changes else 0
            estimated_bpm = avg_changes_per_sec * 60 * 0.8  # Rough BPM estimation
            
            # Build prompt with musical context
            system_prompt = (
                "You are an experienced music teacher providing encouraging feedback on a gesture-based MIDI performance. "
                "Analyze the performance data to identify musical strengths and offer one specific suggestion for growth. "
                "Use accessible musical terminology. Keep response under 100 words. Format as: CHARACTER | STRENGTH | SUGGESTION"
            )
            
            user_prompt = (
                f"### PERFORMANCE METRICS ###\n"
                f"- Note range: {note_range} semitones ({min_note_name} to {max_note_name})\n"
                f"- Average pitch: {midi_note_to_name(int(avg_note))}\n"
                f"- Velocity profile: {int(avg_velocity)} avg, {int(vel_range)} range, {int(vel_stability)}% stability\n"
                f"- Movement pattern: {'upward-dominant' if movement_ratio > 1.2 else 'downward-dominant' if movement_ratio < 0.8 else 'balanced'}\n"
                f"- Estimated tempo: {int(estimated_bpm)} BPM\n\n"
                f"### FEEDBACK INSTRUCTIONS ###\n"
                f"1. CHARACTER: Describe the musical character/mood in 3-5 words\n"
                f"2. STRENGTH: Identify one specific musical strength demonstrated\n"
                f"3. SUGGESTION: Offer one practical suggestion for musical exploration\n"
                f"4. Keep tone encouraging and professional\n"
                f"5. Format strictly as: CHARACTER | STRENGTH | SUGGESTION"
            )
            
            r = self.client.chat.completions.create(
                model="qwen/qwen3-32b",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=120
            )
            txt = r.choices[0].message.content.strip()
            
            # Format and display the feedback
            self._display_musical_feedback(txt)
            
        except Exception as e:
            if self.parent:
                self.parent.update_log(f"LLM Musical Feedback error: {e}")
        finally:
            # Clear log for next session
            self.gesture_log = []

    def _display_musical_feedback(self, feedback_text):
        """Format and display the musical feedback in the UI."""
        if not self.parent:
            return
            
        parts = [p.strip() for p in feedback_text.split("|")]
        
        if len(parts) >= 3:
            character = parts[0]
            strength = parts[1]
            suggestion = parts[2]
            
            # Format with visual separators and emoji for clarity
            header = "🎵 LLM MUSICAL FEEDBACK 🎵"
            formatted_feedback = (
                f"\n{header}\n"
                f"{'─' * len(header)}\n"
                f"Character: {character}\n"
                f"Strength: {strength}\n"
                f"Suggestion: {suggestion}\n"
                f"{'─' * len(header)}"
            )
            self.parent.update_log(formatted_feedback)
            self.parent.update_musical_insight(formatted_feedback)
        else:
            # Fallback if parsing fails - show raw feedback
            clean_text = feedback_text[:150] + "..." if len(feedback_text) > 150 else feedback_text
            self.parent.update_log(f"🎵 LLM Feedback: {clean_text}")
            self.parent.update_musical_insight(f"🎵 LLM Feedback: {clean_text}")

# ------------------------- Video Processing Thread ------------------------
class GestureSynthVideoThread(QThread):
    update_frame = pyqtSignal(QImage)
    update_log = pyqtSignal(str)
    update_note = pyqtSignal(int, int)
    update_hand_state = pyqtSignal(bool, bool)
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.running = False
        self.current_note = None
        self.current_velocity = None
    def stop(self):
        self.running = False
        self.wait()
    def note_name(self, n): return midi_note_to_name(n)
    def run(self):
        # MIDI
        try:
            ports = mido.get_output_names()
            if not ports:
                self.update_log.emit("No MIDI ports available.")
                return
            port_name = self.params['midi_port']
            if port_name not in ports:
                port_name = ports[0]
                self.update_log.emit(f"Selected MIDI not available; using '{port_name}'.")
            self.midi = mido.open_output(port_name)
            self.update_log.emit(f"Connected to MIDI: {port_name}")
        except Exception as e:
            self.update_log.emit(f"MIDI init error: {e}")
            return
        # camera
        cap = cv2.VideoCapture(self.params['camera_index'])
        if not cap.isOpened():
            self.update_log.emit("Could not open webcam.")
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, VIDEO_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, VIDEO_H)
        cap.set(cv2.CAP_PROP_FPS, 30)
        # mediapipe
        mp_hands = mp.solutions.hands
        mp_draw = mp.solutions.drawing_utils
        hands = mp_hands.Hands(min_detection_confidence=0.6,
                               min_tracking_confidence=0.6, max_num_hands=2)
        # smoothing
        note_hist = collections.deque(maxlen=max(1, self.params['smoothing_window']))
        vel_hist = collections.deque(maxlen=max(1, self.params['smoothing_window']))
        chord_map = {'Major':[0,4,7], 'Minor':[0,3,7], '7th':[0,4,7,10]}
        last_note = None
        last_note_sent = None
        last_log_fist = {}
        last_left = False
        last_right = False
        rm = self.params.get('recording_manager', None)
        def hand_to_midi(landmarks):
            hand_y = landmarks.landmark[0].y
            hand_x = landmarks.landmark[0].x
            raw_note = int(np.interp(hand_y, [0,1], [self.params['max_note'], self.params['min_note']]))
            if self.params['velocity_mapping'] == 0:    # Vertical
                raw_vel = int(np.interp(hand_y, [0,1], [127, 30]))
            elif self.params['velocity_mapping'] == 1:  # Horizontal
                raw_vel = int(np.interp(hand_x, [0,1], [127, 30]))
            else:                                       # Fixed
                raw_vel = int(self.params['fixed_velocity'])
            if self.params['note_smoothing']:
                note_hist.append(raw_note); vel_hist.append(raw_vel)
                note = int(sum(note_hist)/len(note_hist))
                vel = int(sum(vel_hist)/len(vel_hist))
            else:
                note, vel = raw_note, raw_vel
            return note, max(30, min(127, vel))
        def is_fist(landmarks):
            tips_mids = [(8,6),(12,10),(16,14),(20,18)]
            return all(landmarks.landmark[t].y > landmarks.landmark[m].y for t,m in tips_mids)
        self.running = True
        time.sleep(0.12)  # let UI finish first paint
        while self.running:
            ok, frame = cap.read()
            if not ok:
                continue
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = hands.process(rgb)
            left_active = False
            right_active = False
            now = time.time()
            if res.multi_hand_landmarks:
                for idx, lm in enumerate(res.multi_hand_landmarks):
                    label = res.multi_handedness[idx].classification[0].label  # 'Left' or 'Right'
                    is_left = (label == "Left")
                    if is_left: left_active = True
                    else: right_active = True
                    # draw landmarks with clear colors (BGR)
                    color_l = (60, 60, 255)   # red-ish
                    color_r = (255, 60, 60)   # blue-ish
                    lc = color_l if is_left else color_r
                    mp_draw.draw_landmarks(
                        frame, lm, mp_hands.HAND_CONNECTIONS,
                        mp_draw.DrawingSpec(color=lc, thickness=2, circle_radius=2),
                        mp_draw.DrawingSpec(color=(200,200,200), thickness=1)
                    )
                    # hand label near wrist
                    wx = int(lm.landmark[0].x * frame.shape[1])
                    wy = int(lm.landmark[0].y * frame.shape[0])
                    txt = "LEFT" if is_left else "RIGHT"
                    cv2.rectangle(frame, (wx-34, wy-26), (wx+42, wy-6), (12,12,12), -1)
                    cv2.putText(frame, txt, (wx-30, wy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, lc, 1, cv2.LINE_AA)
                    # hold logic
                    fist = is_fist(lm)
                    if last_log_fist.get(txt) != fist:
                        self.update_log.emit(f"{txt} {'fist — holding note' if fist else 'open — note can change'}")
                        last_log_fist[txt] = fist
                    note, velocity = hand_to_midi(lm)
                    self.current_note = note
                    self.current_velocity = velocity
                    if not fist:
                        if (last_note is None) or (abs(note - last_note) >= self.params['note_change_threshold']):
                            # turn off previous notes
                            if last_note_sent is not None:
                                try:
                                    if self.params.get('chord_mode', False):
                                        for i in chord_map.get(self.params.get('chord_type','Major'), [0,4,7]):
                                            self.midi.send(mido.Message('note_off', note=last_note_sent+i, channel=0))
                                            if rm: rm.log_event(last_note_sent+i, 0, 'off', 0, now)
                                    else:
                                        self.midi.send(mido.Message('note_off', note=last_note_sent, channel=0))
                                        if rm: rm.log_event(last_note_sent, 0, 'off', 0, now)
                                except Exception:
                                    pass
                            # send new note(s)
                            try:
                                if self.params.get('chord_mode', False):
                                    for i in chord_map.get(self.params.get('chord_type','Major'), [0,4,7]):
                                        self.midi.send(mido.Message('note_on', note=note+i, velocity=velocity, channel=0))
                                        if rm: rm.log_event(note+i, velocity, 'on', 0, now)
                                        time.sleep(0.002)
                                else:
                                    self.midi.send(mido.Message('note_on', note=note, velocity=velocity, channel=0))
                                    if rm: rm.log_event(note, velocity, 'on', 0, now)
                            except Exception:
                                pass
                            last_note = note
                            last_note_sent = note
                            self.update_note.emit(note, velocity)
                    # small HUD
                    hud = f"{txt}: {midi_note_to_name(self.current_note)}  Vel:{int(self.current_velocity/127*100)}%"
                    w = 8*len(hud)+16
                    cv2.rectangle(frame, (12, 12+22*idx), (12+w, 12+22*idx+18), (16,16,16), -1)
                    cv2.putText(frame, hud, (18, 26+22*idx), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230,240,255), 1, cv2.LINE_AA)
            else:
                # if no hands, ensure previous note is turned off
                if last_note_sent is not None:
                    try:
                        if self.params.get('chord_mode', False):
                            for i in chord_map.get(self.params.get('chord_type','Major'), [0,4,7]):
                                self.midi.send(mido.Message('note_off', note=last_note_sent+i, channel=0))
                                if rm: rm.log_event(last_note_sent+i, 0, 'off', 0, time.time())
                        else:
                            self.midi.send(mido.Message('note_off', note=last_note_sent, channel=0))
                            if rm: rm.log_event(last_note_sent, 0, 'off', 0, time.time())
                    except Exception:
                        pass
                    last_note_sent = None
                    self.current_note = None
                    self.current_velocity = None
                    self.update_note.emit(0, 0)
            if (left_active != last_left) or (right_active != last_right):
                self.update_hand_state.emit(left_active, right_active)
                last_left, last_right = left_active, right_active
            # push frame to UI
            qimg = QImage(frame.data, frame.shape[1], frame.shape[0], frame.shape[1]*3, QImage.Format_BGR888)
            self.update_frame.emit(qimg)
            # modest fps to avoid CPU spikes
            time.sleep(1/30)
        cap.release()
        try:
            self.midi.close()
        except Exception:
            pass
# ----------------------------- Note Guide ---------------------------------
class NoteGuideWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(NOTE_GUIDE_W)
        self.setFixedHeight(380)
        self.note_range = (21, 108)
        self.left_hand_active = False
        self.right_hand_active = False
        self.current_note = 0
        self.current_velocity = 0
        self.feedback_mode_active = False
        self.analysis_in_progress = False
        self.recording = False
        
    def set_feedback_mode_active(self, active): self.feedback_mode_active = active; self.update()
    def set_analysis_in_progress(self, val): self.analysis_in_progress = val; self.update()
    def set_recording_state(self, recording): self.recording = recording; self.update()
    
    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        grad = QLinearGradient(0,0,0,self.height())
        grad.setColorAt(0, QColor(22,32,50))
        grad.setColorAt(1, QColor(12,22,42))
        p.fillRect(self.rect(), QBrush(grad))
        
        # Record indicator (pulsing dot when recording)
        if self.recording:
            p.setPen(QPen(QColor(255, 60, 60), 1))
            p.setBrush(QColor(255, 80, 80, 180))
            p.drawEllipse(8, 8, 12, 12)
            p.setPen(QColor(230, 230, 240))
            p.drawText(24, 18, "REC")
        
        # Center line
        p.setPen(QPen(QColor(100,200,120,150), 2))
        p.drawLine(self.width()//2, 28, self.width()//2, self.height()-10)
        
        # Octave ticks with visual hierarchy
        p.setPen(QPen(QColor(180, 180, 190, 90), 1))
        for i in range(0,128,12):
            if self.note_range[0] <= i <= self.note_range[1]:
                t = (i-self.note_range[0])/(self.note_range[1]-self.note_range[0])
                y = int((1-t)*(self.height()-40))+20
                p.drawLine(10, y, self.width()-10, y)
                if i % 24 == 0:
                    p.setPen(QColor(205,205,210))
                    p.drawText(12, y-2, midi_note_to_name(i))
                    p.setPen(QPen(QColor(180,180,190,90),1))
        
        # Current note pointer with animation
        if self.current_note:
            t = (self.current_note-self.note_range[0])/(self.note_range[1]-self.note_range[0])
            y = int((1-t)*(self.height()-40))+20
            p.setPen(QPen(QColor(255,255,90), 2))
            p.setBrush(QColor(255,255,90,110))
            p.drawEllipse(self.width()//2-12, y-12, 24, 24)
            p.setPen(QColor(235,240,255))
            p.drawText(self.width()//2-32, y+4, midi_note_to_name(self.current_note))
        
        # Hand badges with distinct colors
        badge_height = 22
        badge_width = 95
        badge_radius = 8
        
        if self.left_hand_active:
            p.setBrush(QColor(220,70,70,190))
            p.setPen(QPen(QColor(180,40,40),1))
            p.drawRoundedRect(8, self.height()-70, badge_width, badge_height, badge_radius, badge_radius)
            p.setPen(QColor(255,255,255))
            p.drawText(14, self.height()-53, "LEFT HAND")
        
        if self.right_hand_active:
            p.setBrush(QColor(70,70,220,190))
            p.setPen(QPen(QColor(40,40,180),1))
            p.drawRoundedRect(8, self.height()-46, badge_width, badge_height, badge_radius, badge_radius)
            p.setPen(QColor(255,255,255))
            p.drawText(14, self.height()-29, "RIGHT HAND")
        
        # LLM Analysis indicator
        if self.analysis_in_progress:
            p.setPen(QColor(0,255,255))
            p.drawText(self.width()//2-38, self.height()-8, "LLM Analysis")
# -------------------------- Performance Display ---------------------------
class PerformanceDisplay(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(120)
        self._current_velocity = 0
        self.current_note = 0
        self.recording = False
        self.note_animation = 0.0
        
        # Create animation for smooth note transitions
        from PyQt5.QtCore import QPropertyAnimation
        self.anim = QPropertyAnimation(self, b"current_velocity")
        self.anim.setDuration(220)
        self.anim.setEasingCurve(QEasingCurve.OutCubic)
        
    @pyqtProperty(float)
    def note_transition(self):
        return self.note_animation
        
    @note_transition.setter
    def note_transition(self, value):
        self.note_animation = value
        self.update()
        
    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        
        # Background gradient
        g = QLinearGradient(0,0,0,self.height())
        g.setColorAt(0, QColor(15,25,45))
        g.setColorAt(1, QColor(10,20,40))
        p.fillRect(self.rect(), QBrush(g))
        
        # Recording indicator
        if self.recording:
            p.setPen(QPen(QColor(255, 60, 60), 2))
            p.setBrush(QColor(255, 80, 80, 100))
            p.drawEllipse(12, 12, 20, 20)
            p.setPen(QColor(255, 220, 220))
            p.drawText(17, 26, "REC")
        
        # Note name - large and centered
        name = midi_note_to_name(self.current_note)
        p.setPen(QColor(220, 245, 255))
        f = QFont()
        f.setPointSize(24)
        f.setBold(True)
        p.setFont(f)
        txt = f"{name}"
        w = p.fontMetrics().width(txt)
        p.drawText((self.width()-w)//2, 45, txt)
        
        # Velocity label
        p.setPen(QColor(180, 220, 240))
        f_small = QFont()
        f_small.setPointSize(12)
        p.setFont(f_small)
        p.drawText((self.width())//2 - 60, 70, "Velocity")
        
        # Velocity bar container
        vw = int(self.width()*0.8)
        vh = 24
        vx = (self.width()-vw)//2
        vy = 75
        
        # Container styling
        p.setPen(QPen(QColor(70,90,120), 2))
        p.setBrush(QColor(30,45,70))
        p.drawRoundedRect(vx, vy, vw, vh, 12, 12)
        
        # Filled part with gradient
        fill = int(vw * (self._current_velocity/127.0))
        
        # Velocity gradient (green to yellow to red)
        vel_gradient = QLinearGradient(vx, vy, vx+fill, vy+vh)
        if self._current_velocity < 60:
            vel_gradient.setColorAt(0, QColor(70, 170, 70))
            vel_gradient.setColorAt(1, QColor(90, 200, 90))
        elif self._current_velocity < 100:
            vel_gradient.setColorAt(0, QColor(180, 180, 50))
            vel_gradient.setColorAt(1, QColor(220, 220, 80))
        else:
            vel_gradient.setColorAt(0, QColor(200, 60, 60))
            vel_gradient.setColorAt(1, QColor(240, 80, 80))
            
        p.setBrush(vel_gradient)
        p.setPen(Qt.NoPen)
        p.drawRoundedRect(vx, vy, max(4, fill), vh, 12, 12)
        
        # Velocity percentage - inside the bar with contrasting text
        p.setPen(QColor(235,245,255))
        f_percent = QFont()
        f_percent.setPointSize(12)
        f_percent.setBold(True)
        p.setFont(f_percent)
        pct_text = f"{int(self._current_velocity/127.0*100)}%"
        tw = p.fontMetrics().width(pct_text)
        p.drawText(vx + (fill - tw)//2, vy + vh - 6, pct_text)
        
    @pyqtProperty(int)
    def current_velocity(self): 
        return self._current_velocity
        
    @current_velocity.setter
    def current_velocity(self, v): 
        self._current_velocity = int(v)
        self.update()
        
    def update_performance(self, note, velocity, recording=False):
        self.current_note = note
        self.recording = recording
        self.anim.stop()
        self.anim.setStartValue(self._current_velocity)
        self.anim.setEndValue(int(velocity))
        self.anim.start()
        self.update()
# ------------------------------- Controls ---------------------------------
class ControlPanel(QWidget):
    feedback_mode_changed = pyqtSignal(bool)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.initUI()
        
    def initUI(self):
        main = QVBoxLayout(self)
        main.setSpacing(12)
        main.setContentsMargins(14, 12, 14, 12)
        
        # Status indicators at the top
        status_row = QHBoxLayout()
        
        # Camera status indicator
        camera_status = QLabel()
        camera_status.setFixedSize(20, 20)
        camera_status.setStyleSheet("""
            background-color: #4CAF50;
            border-radius: 10px;
            border: 1px solid #388E3C;
        """)
        camera_status.setToolTip("Camera connected")
        
        # MIDI status indicator
        midi_status = QLabel()
        midi_status.setFixedSize(20, 20)
        midi_status.setStyleSheet("""
            background-color: #4CAF50;
            border-radius: 10px;
            border: 1px solid #388E3C;
        """)
        midi_status.setToolTip("MIDI device connected")
        
        status_row.addWidget(QLabel("Status:"))
        status_row.addWidget(camera_status)
        status_row.addWidget(QLabel("Camera"))
        status_row.addWidget(midi_status)
        status_row.addWidget(QLabel("MIDI"))
        status_row.addStretch()
        main.addLayout(status_row)
        
        # Connection
        conn = QGroupBox("Connection Settings")
        conn.setStyleSheet("""
            QGroupBox {
                border: 1px solid #4a90e2;
                border-radius: 6px;
                margin-top: 1ex;
                font-weight: bold;
                color: #4a90e2;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 8px;
                padding: 0 4px;
                background-color: rgba(74, 144, 226, 0.2);
            }
        """)
        grid = QGridLayout()
        grid.setSpacing(12)
        conn.setLayout(grid)
        
        # MIDI Output with visual indicator
        midi_label = QLabel("MIDI Output:")
        midi_label.setStyleSheet("font-weight: bold;")
        self.midi_combo = QComboBox()
        self.midi_combo.setMinimumHeight(30)
        self.midi_combo.setMinimumWidth(240)
        
        try:
            ports = mido.get_output_names()
            self.midi_combo.addItems(ports if ports else ["No MIDI ports available"])
            if not ports: 
                self.midi_combo.setEnabled(False)
                midi_status.setStyleSheet("""
                    background-color: #f44336;
                    border-radius: 10px;
                    border: 1px solid #d32f2f;
                """)
                midi_status.setToolTip("No MIDI devices available")
        except Exception as e:
            self.midi_combo.addItem(f"Error loading MIDI ports: {e}")
            self.midi_combo.setEnabled(False)
            midi_status.setStyleSheet("""
                background-color: #f44336;
                border-radius: 10px;
                border: 1px solid #d32f2f;
            """)
            midi_status.setToolTip(f"MIDI error: {e}")
            
        # Webcam with visual indicator
        cam_label = QLabel("Webcam:")
        cam_label.setStyleSheet("font-weight: bold;")
        self.camera_combo = QComboBox()
        self.camera_combo.setMinimumHeight(30)
        self.camera_combo.setMinimumWidth(240)
        
        cams = []
        for i in range(5):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                cams.append(f"Camera {i}")
                cap.release()
                
        self.camera_combo.addItems(cams if cams else ["No cameras available"])
        if not cams: 
            self.camera_combo.setEnabled(False)
            camera_status.setStyleSheet("""
                background-color: #f44336;
                border-radius: 10px;
                border: 1px solid #d32f2f;
            """)
            camera_status.setToolTip("No cameras available")
            
        grid.addWidget(midi_label, 0, 0)
        grid.addWidget(self.midi_combo, 0, 1)
        grid.addWidget(cam_label, 1, 0)
        grid.addWidget(self.camera_combo, 1, 1)
        main.addWidget(conn)
        
        # Settings
        grp = QGroupBox("Full Control Settings")
        grp.setStyleSheet("""
            QGroupBox {
                border: 1px solid #4CAF50;
                border-radius: 6px;
                margin-top: 1ex;
                font-weight: bold;
                color: #4CAF50;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 8px;
                padding: 0 4px;
                background-color: rgba(76, 175, 80, 0.2);
            }
        """)
        lay = QGridLayout()
        lay.setSpacing(12)
        grp.setLayout(lay)
        
        # Create bold labels for all settings
        def create_bold_label(text):
            label = QLabel(text)
            label.setStyleSheet("font-weight: bold;")
            return label
        
        # UI Elements
        self.min_note_slider = QSlider(Qt.Horizontal)
        self.min_note_slider.setRange(0, 127)
        self.min_note_slider.setValue(21)
        self.min_note_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                border: 1px solid #999999;
                height: 8px;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #4a4a4a, stop:1 #2a2a2a);
                margin: 2px 0;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: qradialgradient(cx:0.5, cy:0.5, radius:0.5, fx:0.5, fy:0.5, stop:0 #4CAF50, stop:1 #2E7D32);
                border: 1px solid #2E7D32;
                width: 18px;
                height: 18px;
                margin: -5px 0;
                border-radius: 9px;
            }
        """)
        
        self.max_note_slider = QSlider(Qt.Horizontal)
        self.max_note_slider.setRange(0, 127)
        self.max_note_slider.setValue(108)
        self.max_note_slider.setStyleSheet(self.min_note_slider.styleSheet())
        
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setRange(1, 20)
        self.threshold_slider.setValue(6)
        self.threshold_slider.setStyleSheet(self.min_note_slider.styleSheet())
        
        self.smoothing_slider = QSlider(Qt.Horizontal)
        self.smoothing_slider.setRange(1, 20)
        self.smoothing_slider.setValue(8)
        self.smoothing_slider.setStyleSheet(self.min_note_slider.styleSheet())
        
        self.velocity_combo = QComboBox()
        self.velocity_combo.addItems(["Vertical Position","Horizontal Position","Fixed Value"])
        self.velocity_combo.setMinimumHeight(30)
        
        self.velocity_spin = QDoubleSpinBox()
        self.velocity_spin.setRange(30, 127)
        self.velocity_spin.setValue(64)
        self.velocity_spin.setMinimumHeight(30)
        
        self.orientation_slider = QSlider(Qt.Horizontal)
        self.orientation_slider.setRange(0, 100)
        self.orientation_slider.setValue(70)
        self.orientation_slider.setStyleSheet(self.min_note_slider.styleSheet())
        
        self.smoothing_checkbox = QCheckBox("Enable Note Smoothing")
        self.smoothing_checkbox.setChecked(True)
        self.smoothing_checkbox.setStyleSheet("font-weight: bold;")
        
        self.chord_checkbox = QCheckBox("Enable Chord Mode")
        self.chord_checkbox.setChecked(False)
        self.chord_checkbox.setStyleSheet("font-weight: bold;")
        
        self.chord_type_combo = QComboBox()
        self.chord_type_combo.addItems(["Major","Minor","7th"])
        self.chord_type_combo.setEnabled(False)
        self.chord_type_combo.setMinimumHeight(30)
        
        # Updated checkbox text
        self.feedback_checkbox = QCheckBox("Enable LLM Musical Feedback")
        self.feedback_checkbox.setChecked(False)
        self.feedback_checkbox.setStyleSheet("font-weight: bold;")
        
        self.min_note_value = QLabel("21 (A0)")
        self.max_note_value = QLabel("108 (C8)")
        self.threshold_value = QLabel("6")
        self.smoothing_value = QLabel("8")
        self.orientation_value = QLabel("70%")
        
        # Bold labels for all settings
        lay.addWidget(create_bold_label("Min Note:"), 0, 0)
        lay.addWidget(self.min_note_slider, 0, 1)
        lay.addWidget(self.min_note_value, 0, 2)
        
        lay.addWidget(create_bold_label("Max Note:"), 1, 0)
        lay.addWidget(self.max_note_slider, 1, 1)
        lay.addWidget(self.max_note_value, 1, 2)
        
        lay.addWidget(create_bold_label("Note Change Threshold:"), 2, 0)
        lay.addWidget(self.threshold_slider, 2, 1)
        lay.addWidget(self.threshold_value, 2, 2)
        
        lay.addWidget(create_bold_label("Smoothing Window:"), 3, 0)
        lay.addWidget(self.smoothing_slider, 3, 1)
        lay.addWidget(self.smoothing_value, 3, 2)
        
        lay.addWidget(create_bold_label("Velocity Mapping:"), 4, 0)
        lay.addWidget(self.velocity_combo, 4, 1)
        
        lay.addWidget(create_bold_label("Fixed Velocity:"), 5, 0)
        lay.addWidget(self.velocity_spin, 5, 1)
        
        lay.addWidget(create_bold_label("Orientation Sensitivity:"), 6, 0)
        lay.addWidget(self.orientation_slider, 6, 1)
        lay.addWidget(self.orientation_value, 6, 2)
        
        lay.addWidget(self.smoothing_checkbox, 7, 0, 1, 3)
        lay.addWidget(self.chord_checkbox, 8, 0, 1, 3)
        lay.addWidget(create_bold_label("Chord Type:"), 9, 0)
        lay.addWidget(self.chord_type_combo, 9, 1)
        lay.addWidget(self.feedback_checkbox, 10, 0, 1, 3)
        
        main.addWidget(grp)
        
        # Recording section
        rec_group = QGroupBox("Recording Controls")
        rec_group.setStyleSheet("""
            QGroupBox {
                border: 1px solid #FF5722;
                border-radius: 6px;
                margin-top: 1ex;
                font-weight: bold;
                color: #FF5722;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 8px;
                padding: 0 4px;
                background-color: rgba(255, 87, 34, 0.2);
            }
        """)
        rec_layout = QHBoxLayout()
        rec_layout.setContentsMargins(10, 15, 10, 10)
        rec_group.setLayout(rec_layout)
        
        # Recording controls
        self.record_button = QPushButton("Start Recording")
        self.record_button.setCheckable(True)
        self.record_button.setMinimumHeight(35)
        self.record_button.setStyleSheet("""
            QPushButton {
                background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #FF7043, stop:1 #E64A19);
                color: white;
                font-weight: bold;
                border-radius: 5px;
                border: 1px solid #D84315;
                font-size: 13px;
                padding: 5px 15px;
            }
            QPushButton:hover {
                background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #FF8A65, stop:1 #FF5722);
            }
            QPushButton:pressed {
                background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #E64A19, stop:1 #BF360C);
            }
            QPushButton:checked {
                background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #f44336, stop:1 #d32f2f);
                border: 1px solid #c62828;
            }
        """)
        
        self.download_button = QPushButton("Download Recording")
        self.download_button.setEnabled(False)
        self.download_button.setMinimumHeight(35)
        self.download_button.setStyleSheet("""
            QPushButton {
                background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #90CAF9, stop:1 #42A5F5);
                color: white;
                font-weight: bold;
                border-radius: 5px;
                border: 1px solid #1E88E5;
                font-size: 13px;
                padding: 5px 15px;
            }
            QPushButton:hover {
                background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #90CAF9, stop:1 #64B5F6);
            }
            QPushButton:pressed {
                background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #42A5F5, stop:1 #1E88E5);
            }
            QPushButton:disabled {
                background-color: #78909C;
                border: 1px solid #546E7A;
            }
        """)
        
        rec_layout.addWidget(self.record_button)
        rec_layout.addWidget(self.download_button)
        main.addWidget(rec_group)
        
        # Run/stop controls - made larger and more prominent
        control_layout = QHBoxLayout()
        
        self.run_button = QPushButton("START ▶")
        self.run_button.setMinimumHeight(45)
        self.run_button.setStyleSheet("""
            QPushButton {
                background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #4CAF50, stop:1 #2E7D32);
                color: white;
                font-weight: bold;
                font-size: 15px;
                border-radius: 8px;
                border: 2px solid #2E7D32;
                padding: 5px 20px;
            }
            QPushButton:hover {
                background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #66BB6A, stop:1 #43A047);
                border: 2px solid #388E3C;
            }
            QPushButton:pressed {
                background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #388E3C, stop:1 #1B5E20);
            }
            QPushButton:disabled {
                background-color: #78909C;
                border: 2px solid #607D8B;
            }
        """)
        
        self.stop_button = QPushButton("STOP ⏹")
        self.stop_button.setMinimumHeight(45)
        self.stop_button.setEnabled(False)
        self.stop_button.setStyleSheet("""
            QPushButton {
                background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #f44336, stop:1 #d32f2f);
                color: white;
                font-weight: bold;
                font-size: 15px;
                border-radius: 8px;
                border: 2px solid #c62828;
                padding: 5px 20px;
            }
            QPushButton:hover {
                background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #EF5350, stop:1 #E53935);
                border: 2px solid #c62828;
            }
            QPushButton:pressed {
                background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #c62828, stop:1 #b71c1c);
            }
            QPushButton:disabled {
                background-color: #78909C;
                border: 2px solid #607D8B;
            }
        """)
        
        control_layout.addWidget(self.run_button)
        control_layout.addWidget(self.stop_button)
        main.addLayout(control_layout)
        
        # Make the control panel slightly narrower
        self.setMaximumWidth(LEFT_COLUMN_MAX_W - 20)
        
        # Wire signals
        self.min_note_slider.valueChanged.connect(self._min)
        self.max_note_slider.valueChanged.connect(self._max)
        self.threshold_slider.valueChanged.connect(lambda v: self.threshold_value.setText(str(v)))
        self.smoothing_slider.valueChanged.connect(lambda v: self.smoothing_value.setText(str(v)))
        self.orientation_slider.valueChanged.connect(lambda v: self.orientation_value.setText(f"{v}%"))
        self.chord_checkbox.stateChanged.connect(lambda s: self.chord_type_combo.setEnabled(s == Qt.Checked))
        self.feedback_checkbox.stateChanged.connect(lambda s: self.feedback_mode_changed.emit(s == Qt.Checked))
        
        # Connect to parent handlers
        self.record_button.toggled.connect(lambda checked: self.parent.toggle_recording(checked))
        self.download_button.clicked.connect(lambda: self.parent.download_recording())
    
    def _min(self, v):
        self.min_note_value.setText(f"{v} ({midi_note_to_name(v)})")
    
    def _max(self, v):
        self.max_note_value.setText(f"{v} ({midi_note_to_name(v)})")

# ------------------------------- Main UI with Logo ----------------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("GestureSynth")
        icon_path = os.path.join(script_dir, 'icon.ico')
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))
            
        self.setStyleSheet("""
            QMainWindow { 
                background-color: #152030; 
            }
            QLabel { 
                color: #cfe3f0; 
                font-size: 13px; 
            }
            QGroupBox { 
                color: #cfe3f0; 
                font-size: 15px; 
            }
        """)
        cw = QWidget()
        self.setCentralWidget(cw)
        root = QVBoxLayout(cw)
        root.setContentsMargins(20, 12, 20, 20)
        root.setSpacing(16)
        
        # header
        top = QHBoxLayout()
        title = QLabel("GestureSynth — Hand-Gesture MIDI")
        title.setStyleSheet("font-size: 20px; font-weight: 700; color: #4cc9ff;")
        top.addWidget(title)
        top.addSpacerItem(QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))
        root.addLayout(top)
        
        # content
        content = QHBoxLayout()
        content.setSpacing(18)
        
        # LEFT column (compact)
        left = QVBoxLayout()
        left.setSpacing(14)
        self.control_panel = ControlPanel(self)
        self.control_panel.setMaximumWidth(LEFT_COLUMN_MAX_W)
        left.addWidget(self.control_panel)
        
        # performance + log
        self.perf = PerformanceDisplay(self)
        left.addWidget(self.perf)
        
        log_group = QGroupBox("System Log")
        l = QVBoxLayout(log_group)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet("QTextEdit { background:#102036; color:#d7e6f5; font-family: Consolas; font-size:12px; }")
        l.addWidget(self.log_text)
        left.addWidget(log_group)
        
        content.addLayout(left, 1)
        
        # RIGHT column (note guide + video)
        right = QVBoxLayout()
        right.setSpacing(12)
        band = QHBoxLayout()
        band.setSpacing(16)
        
        # note guide – narrow
        self.note_guide = NoteGuideWidget(self)
        self.note_guide.setFixedWidth(NOTE_GUIDE_W)
        band.addWidget(self.note_guide, 0)
        
        # video group – expanded to fill more space
        video_group = QGroupBox("Gesture Input")
        vlay = QVBoxLayout(video_group)
        vlay.setContentsMargins(10, 15, 10, 10)
        
        # Video label - responsive size
        self.video_label = QLabel()
        self.video_label.setMinimumSize(400, 225)  # Minimum size but flexible
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("""
            border: 2px solid #4a6fa5;
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #0e1b2e, stop:1 #152840);
            border-radius: 8px;
        """)
        vlay.addWidget(self.video_label)
        
        # Add flexible space to push video to top
        vlay.addStretch(1)
        
        # Add status hint
        status_hint = QLabel("Position hand in view area")
        status_hint.setStyleSheet("""
            color: #a8b8e0;
            font-size: 12px;
            font-style: italic;
            background-color: rgba(30, 40, 65, 180);
            padding: 3px 8px;
            border-radius: 6px;
            margin-top: 4px;
        """)
        status_hint.setAlignment(Qt.AlignCenter)
        vlay.addWidget(status_hint)
        
        band.addWidget(video_group, 1)
        
        right.addLayout(band)
        content.addLayout(right, 2)
        root.addLayout(content)
        
        # services
        self.video_thread = None
        self.recording_manager = RecordingManager(self)
        self.adaptive_mode = AdaptiveModeManager(self, None)
        
        # connect controls
        self.control_panel.run_button.clicked.connect(self.start_processing)
        self.control_panel.stop_button.clicked.connect(self.stop_processing)
        self.control_panel.feedback_mode_changed.connect(self._toggle_feedback_mode)

    def _toggle_feedback_mode(self, enabled):
        """Toggle the feedback mode (which provides automatic parameter adjustments)."""
        self.adaptive_mode.is_active = enabled
        self.note_guide.set_feedback_mode_active(enabled)
        self.update_log(f"Adaptive Mode {'enabled' if enabled else 'disabled'}")

    # ---- control handlers
    def _toggle_adaptive_mode(self, enabled):
        self.adaptive_mode.is_active = enabled
        self.note_guide.set_adaptive_mode_active(enabled)
        self.update_log(f"Adaptive Mode {'enabled' if enabled else 'disabled'}")

    def toggle_recording(self, checked):
        if checked:
            self.recording_manager.start_recording()
            self.control_panel.record_button.setText("Stop Recording")
            self.control_panel.download_button.setEnabled(False)
        else:
            self.recording_manager.stop_recording()
            self.control_panel.record_button.setText("Start Recording")
            self.control_panel.download_button.setEnabled(self.recording_manager.has_events())

    def download_recording(self):
        if not self.recording_manager.has_events():
            QMessageBox.information(self, "No Recording", "There is no recording to download.")
            return
        path = self.recording_manager.save_to_midi(self)
        if path:
            QMessageBox.information(self, "Saved", f"Recording saved to:\n{path}")

    def start_processing(self):
        sel_port = self.control_panel.midi_combo.currentText()
        if sel_port == "No MIDI ports available":
            self.update_log("Cannot start: No MIDI port selected.")
            return
        if sel_port not in mido.get_output_names():
            self.update_log(f"Port '{sel_port}' no longer available.")
            return
            
        params = {
            'midi_port': sel_port,
            'camera_index': self.control_panel.camera_combo.currentIndex(),
            'min_note': self.control_panel.min_note_slider.value(),
            'max_note': self.control_panel.max_note_slider.value(),
            'smoothing_window': self.control_panel.smoothing_slider.value(),
            'note_change_threshold': self.control_panel.threshold_slider.value(),
            'velocity_mapping': self.control_panel.velocity_combo.currentIndex(),
            'fixed_velocity': self.control_panel.velocity_spin.value(),
            'orientation_sensitivity': self.control_panel.orientation_slider.value() / 100.0,
            'note_smoothing': self.control_panel.smoothing_checkbox.isChecked(),
            'chord_mode': self.control_panel.chord_checkbox.isChecked(),
            'chord_type': self.control_panel.chord_type_combo.currentText(),
            'recording_manager': self.recording_manager
        }
        
        if params['min_note'] >= params['max_note']:
            self.update_log("Error: Min Note must be less than Max Note.")
            return
            
        self.video_thread = GestureSynthVideoThread(params)
        self.video_thread.update_frame.connect(self.update_video)
        self.video_thread.update_log.connect(self.update_log)
        self.video_thread.update_note.connect(self.update_note)
        self.video_thread.update_hand_state.connect(self.update_hand_state)
        self.video_thread.start()
        self.control_panel.run_button.setEnabled(False)
        self.control_panel.stop_button.setEnabled(True)
        self.update_log(f"Started. Range {params['min_note']}–{params['max_note']}, smoothing={params['smoothing_window']}.")

    def stop_processing(self):
        if self.video_thread:
            self.video_thread.stop()
            self.video_thread = None
            self.update_log("Stopped.")
            
            # Automatically analyze and adjust parameters if Adaptive Mode is active
            if self.adaptive_mode.is_active and self.adaptive_mode.gesture_log:
                self.adaptive_mode.analyze_and_adjust()
                
            self.control_panel.run_button.setEnabled(True)
            self.control_panel.stop_button.setEnabled(False)

    # ---- UI updates (main thread)
    def update_video(self, img: QImage):
        if self.video_label is None:
            return
        pix = QPixmap.fromImage(img)
        # Scale while maintaining aspect ratio
        scaled = pix.scaled(
            self.video_label.size(), 
            Qt.KeepAspectRatio, 
            Qt.SmoothTransformation
        )
        self.video_label.setPixmap(scaled)

    def update_log(self, msg: str):
        self.log_text.append(msg)
        self.log_text.verticalScrollBar().setValue(self.log_text.verticalScrollBar().maximum())
        logging.info(msg)

    def update_hand_state(self, left_active, right_active):
        self.note_guide.left_hand_active = left_active
        self.note_guide.right_hand_active = right_active
        self.note_guide.update()

    def update_note(self, note, velocity):
        self.note_guide.current_note = note
        self.note_guide.current_velocity = velocity
        self.note_guide.update()
        self.perf.update_performance(note, velocity)
        if self.adaptive_mode.is_active:
            self.adaptive_mode.add_gesture_data(note, velocity)

    def closeEvent(self, e):
        if self.video_thread:
            self.video_thread.stop()
        e.accept()

# --------------------------- Enhanced Adaptive Mode --------------------------------
class AdaptiveModeManager:
    """Intelligent parameter adjustment system with proper evaluation framework."""
    
    # Add historical tracking
    def __init__(self, parent, video_thread):
        self.parent = parent
        self.video_thread = video_thread
        self.client = None
        self.is_active = False
        self.gesture_log = []
        self.key = os.getenv("GROQ_API_KEY")
        self.session_count = 0
        self.historical_data = {
            'note_stability': [],
            'velocity_consistency': [],
            'note_change_speed': []
        }
        self.optimization_history = []  # Track what changes were made and their impact
        
        # Performance thresholds (personalized by user)
        self.thresholds = {
            'note_stability': 0.8,  # 80% stability is considered good
            'velocity_consistency': 0.7,  # 70% consistency is good
            'note_change_speed': 3.0  # 3 semitones/second is ideal for most users
        }
        
        # Initialize with default values
        self.current_settings = {
            'smoothing_window': 8,
            'note_change_threshold': 6,
            'velocity_mapping': 'Vertical Position'
        }
        
        if self.key and Groq is not None:
            try:
                self.client = Groq(api_key=self.key)
                if self.parent:
                    self.parent.update_log("Adaptive Mode ready (disabled by default).")
            except Exception as e:
                if self.parent:
                    self.parent.update_log(f"Adaptive Mode: could not init Groq: {e}")
        else:
            if self.parent:
                self.parent.update_log("Adaptive Mode ready (disabled by default).")

    def add_gesture_data(self, note, velocity, timestamp=None):
        """Collect gesture data only when Adaptive Mode is active."""
        if not self.is_active:
            return
        if timestamp is None:
            timestamp = time.time()
        self.gesture_log.append({"timestamp": timestamp, "note": note, "velocity": velocity})
        if len(self.gesture_log) > 500:
            self.gesture_log = self.gesture_log[-500:]

    def analyze_and_adjust(self):
        """Analyze gesture patterns and determine if adjustments are needed."""
        if not self.is_active or not self.client or len(self.gesture_log) < 50:
            return
            
        try:
            self.parent.update_log("Adaptive Mode: analyzing performance patterns...")
            metrics = self._extract_performance_metrics()
            
            # Calculate performance scores
            performance_scores = self._calculate_performance_scores(metrics)
            
            # Determine if adjustments are needed
            need_adjustment, reasons = self._evaluate_needs_adjustment(performance_scores)
            
            if need_adjustment:
                # Get specific adjustments from LLM
                adjustments = self._get_optimal_adjustments(metrics, performance_scores)
                if adjustments:
                    self._apply_adjustments(adjustments)
                    self._report_changes(adjustments, metrics, reasons)
                else:
                    self.parent.update_log("Adaptive Mode: analysis complete but no specific improvements were identified.")
            else:
                self._report_optimal_settings(performance_scores, reasons)
                
            # Record this session for historical context
            self._update_historical_data(metrics, performance_scores)
            
        except Exception as e:
            self.parent.update_log(f"Adaptive Mode error: {e}")
        finally:
            self.gesture_log = []  # Clear after analysis

    def _extract_performance_metrics(self):
        """Calculate meaningful metrics from gesture data with historical context."""
        notes = [x["note"] for x in self.gesture_log]
        velocities = [x["velocity"] for x in self.gesture_log]
        timestamps = [x["timestamp"] for x in self.gesture_log]
        
        # Note stability metrics
        note_changes = []
        for i in range(1, len(notes)):
            change = abs(notes[i] - notes[i-1])
            if change > 0:
                time_diff = timestamps[i] - timestamps[i-1]
                note_changes.append((change, time_diff))
        
        rapid_changes = sum(1 for c, t in note_changes if c > 10 and t < 0.2)
        total_changes = len(note_changes)
        note_stability = 1.0 - (rapid_changes / max(1, total_changes))
        
        # Velocity consistency
        vel_std = np.std(velocities) if len(velocities) > 1 else 0
        vel_range = max(velocities) - min(velocities)
        vel_consistency = 1.0 - (vel_std / 127.0) if vel_range > 0 else 0.0
        
        # Note change speed
        avg_change_speed = 0.0
        if note_changes:
            avg_change_speed = sum(c/t for c, t in note_changes if t > 0) / len(note_changes)
        
        return {
            'note_range': max(notes) - min(notes),
            'rapid_note_changes': rapid_changes,
            'total_note_changes': total_changes,
            'note_stability': note_stability,
            'velocity_std': vel_std,
            'velocity_range': vel_range,
            'velocity_consistency': vel_consistency,
            'avg_change_speed': avg_change_speed,
            'sample_count': len(self.gesture_log)
        }

    def _calculate_performance_scores(self, metrics):
        """Calculate normalized performance scores (0-1) for each metric."""
        scores = {
            'note_stability': min(1.0, max(0.0, metrics['note_stability'])),
            'velocity_consistency': min(1.0, max(0.0, metrics['velocity_consistency'])),
            'note_change_speed': min(1.0, max(0.0, metrics['avg_change_speed'] / 10.0))  # Normalize to 0-1
        }
        
        # Add historical context to scores
        if self.historical_data['note_stability']:
            historical_avg = sum(self.historical_data['note_stability']) / len(self.historical_data['note_stability'])
            scores['note_stability'] = (scores['note_stability'] + historical_avg) / 2
            
        return scores

    def _evaluate_needs_adjustment(self, performance_scores):
        """Determine if current settings need adjustment based on scores and thresholds."""
        reasons = []
        need_adjustment = False
        
        # Evaluate note stability
        if performance_scores['note_stability'] < self.thresholds['note_stability']:
            need_adjustment = True
            reasons.append(f"Note stability ({performance_scores['note_stability']:.2f}) is below target ({self.thresholds['note_stability']:.2f})")
        
        # Evaluate velocity consistency
        if performance_scores['velocity_consistency'] < self.thresholds['velocity_consistency']:
            need_adjustment = True
            reasons.append(f"Velocity consistency ({performance_scores['velocity_consistency']:.2f}) is below target ({self.thresholds['velocity_consistency']:.2f})")
        
        # Evaluate note change speed
        if performance_scores['note_change_speed'] < self.thresholds['note_change_speed'] / 10.0:  # Convert threshold to same scale
            need_adjustment = True
            reasons.append(f"Note change speed ({performance_scores['note_change_speed']:.2f}) is below target ({self.thresholds['note_change_speed']:.2f})")
        
        return need_adjustment, reasons

    def _get_optimal_adjustments(self, metrics, performance_scores):
        """Get precise parameter adjustments from LLM with context about current settings."""
        system_prompt = (
            "You are an expert in gesture-to-MIDI mapping optimization. "
            "Analyze performance metrics and recommend EXACT parameter changes only if needed. "
            "If current settings are already optimal, respond with an empty JSON object. "
            "Only respond with valid JSON: {\"parameter\":\"value\"} pairs. "
            "Valid parameters: smoothing_window, note_change_threshold, velocity_mapping"
        )
        
        # Format current settings for the prompt
        current_settings = (
            f"- Smoothing Window: {self.current_settings['smoothing_window']}\n"
            f"- Note Change Threshold: {self.current_settings['note_change_threshold']}\n"
            f"- Velocity Mapping: {self.current_settings['velocity_mapping']}"
        )
        
        # Format performance scores with context
        performance_context = (
            f"- Note stability: {metrics['note_stability']:.2f} (target: {self.thresholds['note_stability']:.2f})\n"
            f"- Velocity consistency: {metrics['velocity_consistency']:.2f} (target: {self.thresholds['velocity_consistency']:.2f})\n"
            f"- Note change speed: {metrics['avg_change_speed']:.1f} semitones/second (target: {self.thresholds['note_change_speed']} semitones/second)"
        )
        
        user_prompt = (
            f"### CURRENT SETTINGS ###\n"
            f"{current_settings}\n\n"
            f"### PERFORMANCE METRICS ###\n"
            f"{performance_context}\n\n"
            f"### INSTRUCTIONS ###\n"
            f"1. If current settings are already optimal (all metrics at or above target), respond with {{}}\n"
            f"2. If specific adjustments would improve performance, respond with ONLY the necessary changes\n"
            f"3. Be specific with values (e.g., 'smoothing_window': 10)\n"
            f"4. Do NOT respond with explanations, only JSON"
        )
        
        try:
            r = self.client.chat.completions.create(
                model="meta-llama/llama-4-scout-17b-16e-instruct",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=100
            )
            response = r.choices[0].message.content.strip()
            
            # Extract JSON from response
            import json
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            if start_idx != -1 and end_idx != -1:
                json_str = response[start_idx:end_idx]
                return json.loads(json_str)
        except Exception as e:
            self.parent.update_log(f"Adaptive Mode: LLM response parsing error: {e}")
        
        return None

    def _apply_adjustments(self, adjustments):
        """Apply parameter adjustments and update current settings."""
        cp = self.parent.control_panel
        applied_changes = []
        
        # Update current settings and apply to UI
        if "smoothing_window" in adjustments:
            new_value = int(adjustments["smoothing_window"])
            new_value = max(1, min(20, new_value))
            cp.smoothing_slider.setValue(new_value)
            self.current_settings['smoothing_window'] = new_value
            applied_changes.append(("Smoothing Window", new_value))
        
        if "note_change_threshold" in adjustments:
            new_value = int(adjustments["note_change_threshold"])
            new_value = max(1, min(20, new_value))
            cp.threshold_slider.setValue(new_value)
            self.current_settings['note_change_threshold'] = new_value
            applied_changes.append(("Note Change Threshold", new_value))
        
        if "velocity_mapping" in adjustments:
            mapping_str = adjustments["velocity_mapping"].lower()
            if "fixed" in mapping_str:
                cp.velocity_combo.setCurrentIndex(2)
                self.current_settings['velocity_mapping'] = "Fixed Value"
                applied_changes.append(("Velocity Mapping", "Fixed Value"))
            elif "horizontal" in mapping_str:
                cp.velocity_combo.setCurrentIndex(1)
                self.current_settings['velocity_mapping'] = "Horizontal Position"
                applied_changes.append(("Velocity Mapping", "Horizontal Position"))
            else:
                cp.velocity_combo.setCurrentIndex(0)
                self.current_settings['velocity_mapping'] = "Vertical Position"
                applied_changes.append(("Velocity Mapping", "Vertical Position"))
        
        return applied_changes

    def _report_changes(self, adjustments, metrics, reasons):
        """Report exactly what changes were made and why."""
        changes = self._apply_adjustments(adjustments)
        if changes:
            report_lines = ["Adaptive Mode: Parameters automatically adjusted:"]
            for param, new_value in changes:
                report_lines.append(f"- {param}: {new_value}")
            
            # Add context about why changes were made
            report_lines.append("\nReasons for adjustment:")
            report_lines.extend([f"- {reason}" for reason in reasons])
            
            # Add performance metrics for context
            report_lines.append("\nCurrent performance metrics:")
            report_lines.append(f"- Note stability: {metrics['note_stability']:.2f}")
            report_lines.append(f"- Velocity consistency: {metrics['velocity_consistency']:.2f}")
            report_lines.append(f"- Note change speed: {metrics['avg_change_speed']:.1f} semitones/second")
            
            self.parent.update_log("\n".join(report_lines))
        else:
            self.parent.update_log("Adaptive Mode: analysis complete but no specific improvements were identified.")

    def _report_optimal_settings(self, performance_scores, reasons):
        """Report that current settings are optimal with performance metrics."""
        report_lines = [
            "Adaptive Mode: current settings are optimal for your performance style!",
            "",
            "Performance metrics:",
            f"- Note stability: {performance_scores['note_stability']:.2f}",
            f"- Velocity consistency: {performance_scores['velocity_consistency']:.2f}",
            f"- Note change speed: {performance_scores['note_change_speed']:.2f}",
            "",
            "Why no changes were needed:"
        ]
        
        # Add specific reasons for optimal settings
        if not reasons:
            report_lines.append("- All metrics meet or exceed your personal targets")
        else:
            for reason in reasons:
                report_lines.append(f"- {reason} (already optimal)")
        
        # Add historical context
        if self.session_count > 0:
            report_lines.append("")
            report_lines.append("Your performance has improved from previous sessions!")
        
        self.parent.update_log("\n".join(report_lines))

    def _update_historical_data(self, metrics, performance_scores):
        """Update historical data for future sessions."""
        self.session_count += 1
        
        # Update historical data
        self.historical_data['note_stability'].append(metrics['note_stability'])
        self.historical_data['velocity_consistency'].append(metrics['velocity_consistency'])
        self.historical_data['note_change_speed'].append(metrics['avg_change_speed'])
        
        # Keep only last 10 sessions for relevance
        for key in self.historical_data:
            if len(self.historical_data[key]) > 10:
                self.historical_data[key] = self.historical_data[key][-10:]

# --------------------------------- Main -----------------------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    
    # Set application font
    font = QFont("Segoe UI", 10)
    app.setFont(font)
    
    # Custom palette with improved contrast
    pal = QPalette()
    pal.setColor(QPalette.Window, QColor(20,30,50))
    pal.setColor(QPalette.WindowText, QColor(220,230,240))
    pal.setColor(QPalette.Base, QColor(15,25,40))
    pal.setColor(QPalette.AlternateBase, QColor(25,35,55))
    pal.setColor(QPalette.Text, QColor(220,230,240))
    pal.setColor(QPalette.Button, QColor(25,35,60))
    pal.setColor(QPalette.ButtonText, QColor(220,230,240))
    pal.setColor(QPalette.BrightText, QColor(255,255,255))
    pal.setColor(QPalette.Highlight, QColor(64,183,139))
    pal.setColor(QPalette.HighlightedText, QColor(0,0,0))
    pal.setColor(QPalette.Disabled, QPalette.Text, QColor(120,120,140))
    pal.setColor(QPalette.Disabled, QPalette.ButtonText, QColor(120,120,140))
    
    app.setPalette(pal)
    
    w = MainWindow()
    w.setMinimumSize(1400, 850)
    w.resize(1480, 880)
    
    # Center window
    screen = app.primaryScreen().availableGeometry()
    w.move((screen.width() - w.width()) // 2, (screen.height() - w.height()) // 2)
    
    w.show()
    sys.exit(app.exec_())