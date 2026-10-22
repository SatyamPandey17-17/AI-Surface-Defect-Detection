"""
Read filtered object-height lines from the Arduino serial port and send
the latest reading to Supabase for the hosted Streamlit app.

Install: python -m pip install pyserial requests
Run:     python serial_uploader.py
"""
import os
import re
import time
from datetime import datetime, timezone

import requests
import serial
from serial.tools import list_ports

SERIAL_PORT = os.getenv("ARDUINO_PORT", "COM3")
BAUD_RATE = 9600
SUPABASE_URL = os.getenv("SUPABASE_URL", "").rstrip("/")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
TABLE = "height_readings"
MIN_INTERVAL_S = 0.5  # at most 2 uploads per second

HEIGHT_PATTERN = re.compile(
    r"Object Height:\s*(-?\d+(?:\.\d+)?)\s*cm",
    re.IGNORECASE,
)


def main():
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise SystemExit(
            "Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY environment variables."
        )

    print("Available serial ports:")
    for port in list_ports.comports():
        print(f"  {port.device}: {port.description}")

    print(f"\nConnecting to Arduino on {SERIAL_PORT} at {BAUD_RATE} baud...")
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=minimal",
    }
    endpoint = f"{SUPABASE_URL}/rest/v1/{TABLE}"
    last_sent = 0.0

    try:
        with serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1) as arduino:
            time.sleep(2)  # Arduino resets when the port opens
            arduino.reset_input_buffer()
            print("Connected. Sending filtered height readings. Press Ctrl+C to stop.")

            while True:
                line = arduino.readline().decode("utf-8", errors="ignore").strip()
                if not line:
                    continue
                print("Arduino:", line)

                match = HEIGHT_PATTERN.search(line)
                if not match:
                    continue

                now = time.monotonic()
                if now - last_sent < MIN_INTERVAL_S:
                    continue
                last_sent = now

                height = max(0.0, float(match.group(1)))
                payload = {
                    "object_height": round(height, 1),
                    "created_at": datetime.now(timezone.utc).isoformat(),
                }
                try:
                    response = requests.post(endpoint, headers=headers, json=payload, timeout=10)
                    response.raise_for_status()
                    print(f"Uploaded object height: {height:.1f} cm")
                except requests.RequestException as exc:
                    print(f"Upload failed: {exc}")

    except serial.SerialException as exc:
        raise SystemExit(
            f"Could not open {SERIAL_PORT}: {exc}\n"
            "Close Arduino Serial Monitor and verify Tools > Port in Arduino IDE."
        )
    except KeyboardInterrupt:
        print("\nStopped by user.")


if __name__ == "__main__":
    main()
