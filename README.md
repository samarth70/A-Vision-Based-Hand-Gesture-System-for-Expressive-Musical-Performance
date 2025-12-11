# GestureSynth – Hand-Gesture MIDI Controller

![GestureSynth Demo](demo.gif)  
*Turn hand gestures into expressive MIDI music using just a webcam.*

GestureSynth is a real-time, vision-based MIDI controller that translates hand movement into musical notes. Using MediaPipe for hand tracking and PyQt5 for the interface, it maps hand position to pitch, velocity, and even chords—routing output to any DAW or virtual instrument.

---

## ✨ Features

- **Real-time gesture-to-MIDI conversion** at 30 FPS
- **Multiple velocity modes**: vertical, horizontal, or fixed
- **Chord mode**: play major, minor, or 7th chords with one hand
- **Automatic parameter tuning** via LLM analysis of your playing style
- **MIDI recording & export** as standard `.mid` files
- **Fully offline processing** – your webcam data never leaves your machine
- **Responsive, HCI-optimized UI** following Fitts’s Law, Hick’s Law, and Gestalt principles

---

## 🖥️ Requirements

- Python 3.8+
- A standard webcam
- MIDI-compatible software or hardware (e.g., Ableton Live, FL Studio, GarageBand, Microsoft GS Wavetable Synth)

---

## 🚀 Installation

### 1. Clone the repository
```bash
git clone https://github.com/your-username/gesturesynth.git
cd gesturesynth
```

---
## 📁 Project Structure

<img width="783" height="233" alt="image" src="https://github.com/user-attachments/assets/5338e346-e6f4-426a-b89c-371d671aae7c" />


---
## 🎯 Use Cases

    Music Education: Visual pitch/velocity feedback for students
    Live Performance: Expressive, wireless gestural control
    Accessibility: Play music without physical instruments
    Prototyping: Test musical ideas through movement
---
## Acknowledgements

    MediaPipe by Google for real-time hand tracking
    Mido for MIDI handling in Python
    Groq for fast LLM inference
    PyQt5 for the cross-platform desktop interface
