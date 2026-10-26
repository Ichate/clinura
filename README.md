# Clinura

Clinura reads your heart rate, heart-rate variability and breathing rate through your webcam,
without touching you. It can also write a shift handover note and answer questions about what
happened during monitoring.


## What you need

- A Mac or PC with a webcam
- Python 3.11 or newer
- Node.js 20 or newer
- A DeepSeek API key, if you want the AI handover and the question box (optional)

## Setup

1. Open a terminal in the Clinura folder.

2. Install the Python packages:

   ```bash
   python3 -m pip install -r requirements.txt
   ```

3. To use the AI features, copy the example settings file:

   ```bash
   cp .env.example .env
   ```

   Then open `.env` and paste your DeepSeek key after `DEEPSEEK_API_KEY=`.

4. Start Clinura:

   ```bash
   python3 main.py
   ```

   The first start takes a minute or two while it downloads what it needs and builds the
   interface.

5. Open **http://127.0.0.1:3000** in your browser.

On a Mac, the first start asks for camera access. If it doesn't, allow it for your terminal in
**System Settings → Privacy & Security → Camera**. Clinura keeps trying and connects as soon as
access is given.

To stop Clinura, press `Ctrl + C` in the terminal.

## Using Clinura

### Monitor

This is the live screen. Sit about an arm's length from the camera, face it, and keep still. After
about 10 seconds your heart rate appears. Heart-rate variability (HRV) appears after about 30
seconds, and breathing rate after about 20.

If a number doesn't appear, the **Signal** box tells you what to fix, usually the lighting or
movement. For the best readings:

- Use soft, even light on your face, with no bright window behind you.
- Keep your head still and don't talk.
- Get closer if your face looks small in the video.

Press **Restart reading** to start the measurement over.

### Handover

Clinura keeps a record of the readings while the camera is running, and notes anything unusual:
heart-rate spikes, very fast or slow breathing, movement, times the person was out of view, and
long stretches with eyes closed.

- **Write handover** turns the chosen time window (1, 6, 12 or 24 hours) into a short note you can
  copy into a patient record. Add a bed number and a patient label first if you like. Without a
  DeepSeek key you still get a basic note built from a template.
- **Ask Clinura** answers questions such as *"Were there any heart-rate spikes while the eyes were
  closed?"*, with exact times. Type a question, or press the microphone and speak.
- **Event log** lists everything Clinura noticed. Add your own notes, such as a medication or a
  nurse call, with the box at the top or the quick buttons.

The AI features send a summary of the readings to DeepSeek. Video never leaves your computer.

### Wellness

- **Check-in**: sit still for 60 seconds to save your resting heart rate, HRV and breathing for
  this time of day.
- **Recovery**: compares your current readings with that saved baseline.
- **Paced breathing**: a guide for slow breathing, five seconds in and five seconds out.
- **Profile**: your name, an optional skin-tone setting that only adjusts the lighting advice, and
  your saved baselines.

### Research

Lets you try measuring from your tongue or palm instead of your face. These are experimental and
not reliable.

## Settings

You can change these in `.env`:

| Setting | What it does | Default |
| --- | --- | --- |
| `DEEPSEEK_API_KEY` | Turns on the AI handover and questions | empty |
| `DEEPSEEK_MODEL` | Which DeepSeek model to use | `deepseek-flash` |
| `CLINURA_PORT` | The port the app runs on | `3000` |
| `CLINURA_CAMERA` | Which camera to use, if you have several (0, 1, 2…) | `0` |

## Your data

Everything runs on your computer. The reading history is saved in `data/telemetry.db` and kept
for seven days. Delete that file to start fresh.
