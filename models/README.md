# MediaPipe Face Landmarker model

`face_landmarker.task` is downloaded from Google’s MediaPipe model storage:

```
https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
```

- Size: 3,758,596 bytes (approx. 3.6 MB)
- SHA-256: `64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff`
- Download/verify: `python scripts/download_model.py`
- License: check the MediaPipe model card and Google’s terms before redistribution. The
  repository does not grant additional rights to the model weights.

The engine expects the file at this path by default. Override it through
`FaceConfig.model_path` if needed.

## Hand Landmarker model

`hand_landmarker.task` is used only by the experimental palm ROI mode.

- Source: `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`
- Size: 7,819,105 bytes (approx. 7.5 MB)
- SHA-256: `fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1`
- Download/verify: `python scripts/download_model.py`
- License: check the MediaPipe model card and Google’s terms before redistribution.
