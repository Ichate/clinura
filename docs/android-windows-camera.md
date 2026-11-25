# Android camera on Windows

Clinura consumes a camera device exposed by Windows. It does not pair with Android directly.
The lowest-friction route is Android's native USB webcam mode, which presents a UVC camera to the
host without installing a webcam app on the phone. This requires Android 14 QPR1 or newer **and**
vendor support for the `DeviceAsWebcam` service/UVC gadget; OS version by itself is insufficient.

## Recommended paths

1. **USB UVC (preferred for lower latency):** connect with a data-capable cable, choose Webcam in
   Android USB preferences, then refresh Clinura's camera list. Pixels document these exact steps
   in [Google's Pixel camera help](https://support.google.com/pixelcamera/answer/14274129). The
   [Android platform documentation](https://source.android.com/docs/core/camera/webcam) describes
   the supported UVC implementation and the vendor build requirements.
2. **Phone Link connected camera (wireless fallback):** link Android from Windows 11
   **Settings → Bluetooth & devices → Mobile devices**, enable **Use as connected camera**, and
   grant camera permission to Link to Windows. Microsoft documents Windows 11+, Android 10+ and
   Link to Windows 1.24022.0+ requirements in [its support article](https://support.microsoft.com/en-us/windows/apps/phonelink/use-your-mobile-device-s-camera).
   The feature first appeared in Microsoft's Windows Insider rollout notes; availability depends
   on the installed Windows components and region/device support ([Windows Insider announcement](https://blogs.windows.com/windows-insider/2024/02/29/ability-to-use-a-mobile-devices-camera-as-a-webcam-on-your-pc-begins-rolling-out-to-windows-insiders/)).

## Limitations and troubleshooting

- For native USB webcam mode, the phone manufacturer must enable Android's UVC gadget implementation.
  If **Webcam** is absent from USB preferences, check for a phone firmware update or use Phone Link.
- Phone Link is not an app-free route: the Android Link to Windows app and camera permission are
  required. A consent prompt may appear when Clinura opens the camera.
- Select the phone's webcam entry in Clinura after pressing **Refresh**. If absent, first confirm it
  is visible in Windows Camera. Clinura cannot capture a device Windows does not enumerate.
- Close other camera clients if the device is busy. An Android app using the camera can interrupt
  its webcam stream. Phone temperature, cable quality, USB mode, driver and advertised formats can
  affect sustained frame rate and delay.
- Clinura requests 1920×1080 at 30 fps, but the device negotiates an advertised supported mode.
  Lower delivered resolution/FPS is expected on some phones.
- USB/UVC avoids the network hop and is the preferred low-latency route. Phone Link uses a wireless
  connection, so radio conditions and Windows' camera bridge affect latency.

## Clinura implementation

On Windows, camera enumeration and capture try **DirectShow first, then Media Foundation** for every
device index. A backend is only accepted once it actually delivers a frame — DirectShow can open a
USB/UVC device and then return zero frames, which previously made the camera look "broken". Clinura
now falls through to Media Foundation in that case, so USB webcams and Android UVC webcam mode work
without manual configuration.

These are computer-side backends; no Android app or Clinura phone component is involved for UVC.
macOS continues using AVFoundation so iPhone Continuity Camera remains available.

## Windows troubleshooting

If a USB camera still does not appear or does not stream:

1. **Confirm Windows sees it.** Open the built-in **Camera** app. If the device is missing there,
   Clinura cannot capture it — fix the driver or cable first.
2. **Check privacy settings.** Windows **Settings → Privacy & security → Camera** must allow camera
   access and desktop app access.
3. **Close other camera clients.** Teams, Zoom, OBS, the Camera app and browser tabs can hold the
   device exclusively.
4. **Try a different USB port / cable.** Prefer a direct port over a hub, and a data-capable cable.
5. **Reconnect in webcam mode.** On Android, re-select **Webcam** in USB preferences after plugging in.
6. **Check the log.** Clinura logs which backend opened the camera (`Camera N opened with backend …`)
   and lists every backend it tried when opening fails.
