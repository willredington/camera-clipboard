#!/usr/bin/env bash
# Run on the Pi as your normal user (it uses sudo where needed): ./install.sh
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
USER_NAME="$(id -un)"

sudo apt-get update
sudo apt-get install -y python3-picamera2 python3-flask python3-gpiozero python3-pil python3-numpy python3-requests python3-venv

# --system-site-packages so the venv sees apt's picamera2/libcamera/gpiozero; OpenCV (4.7+) comes from pip.
python3 -m venv --system-site-packages "$DIR/.venv"
"$DIR/.venv/bin/pip" install -r "$DIR/requirements-pi.txt"
"$DIR/.venv/bin/python" -c "import cv2; cv2.aruco.ArucoDetector; print('OpenCV', cv2.__version__, 'ok')"

[ -f "$DIR/config.toml" ] || cp "$DIR/config.example.toml" "$DIR/config.toml"
if [ ! -f "$DIR/clipcam.env" ]; then  # secrets for auto-capture OCR, read by systemd (EnvironmentFile)
  (umask 077; echo "OPENROUTER_API_KEY=" > "$DIR/clipcam.env")
fi
sudo usermod -aG video,gpio "$USER_NAME"

sed -e "s|@USER@|$USER_NAME|g" -e "s|@DIR@|$DIR|g" "$DIR/clipcam.service" | sudo tee /etc/systemd/system/clipcam.service >/dev/null
sudo systemctl daemon-reload
sudo systemctl enable --now clipcam.service
echo "Installed. Open http://$(hostname).local:8080/  (logs: journalctl -u clipcam -f)"
echo "Edit $DIR/config.toml (marker positions!) then: sudo systemctl restart clipcam"
echo "Auto-capture OCR: put your key in $DIR/clipcam.env and set [auto] enabled = true in config.toml"
