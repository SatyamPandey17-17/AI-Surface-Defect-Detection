from pathlib import Path
import re, shutil, zipfile, textwrap

base = Path("/mnt/data/combined_surface_inspection")
base.mkdir(exist_ok=True)

source_path = Path("/mnt/data/Pasted text(2).txt")
source = source_path.read_text(encoding="utf-8")

# Keep original imports and model/checkpoint helper functions, but rebuild page routing.
helper_end = source.index("# ============================================================\n# 8. MODEL INFORMATION")
helpers = source[:helper_end]

# Remove the original page config and title/caption, as combined app adds its own header.
page_start = helpers.index("# ============================================================\n# 2. STREAMLIT PAGE CONFIGURATION")
hf_start = helpers.index("# ============================================================\n# 3. HUGGING FACE TOKEN")
helpers = helpers[:page_start] + helpers[hf_start:]

# Ensure requests imported; REST calls are used in main app.
helpers = helpers.replace("import torch\n", "import torch\nimport requests\n")

ai_body = source[source.index("# ============================================================\n# 8. MODEL INFORMATION"):]

# Indent the full existing AI section so it only runs when that page is selected.
ai_indented = "\n".join(("    " + line if line else "") for line in ai_body.splitlines())

app = helpers + textwrap.dedent('''

# ============================================================
# COMBINED APP NAVIGATION
# ============================================================

st.set_page_config(
    page_title="Surface Inspection & Object Height",
    page_icon="🔍",
    layout="wide",
)

st.title("🔍 Surface Inspection & Object Height")
st.caption("AI steel-surface inspection plus live Arduino HC-SR04 height measurement.")

page = st.sidebar.radio(
    "Choose project module",
    ["Live Object Height", "AI Surface Defect Detection"],
)

if page == "Live Object Height":
    st.header("Live Object Height")
    st.caption("The laptop-side serial bridge sends filtered sensor readings to the cloud database.")

    @st.cache_data(ttl=2, show_spinner=False)
    def fetch_latest_height():
        """Read the newest sensor reading from Supabase REST API."""
        url = st.secrets.get("SUPABASE_URL", "").rstrip("/")
        key = st.secrets.get("SUPABASE_SERVICE_ROLE_KEY", "")
        if not url or not key:
            return None, "Add SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY in Streamlit Secrets."
        endpoint = f"{url}/rest/v1/height_readings"
        headers = {
            "apikey": key,
            "Authorization": f"Bearer {key}",
        }
        params = {
            "select": "object_height,created_at",
            "order": "created_at.desc",
            "limit": "1",
        }
        try:
            response = requests.get(endpoint, headers=headers, params=params, timeout=8)
            response.raise_for_status()
            rows = response.json()
            if not rows:
                return None, "No sensor readings have arrived yet. Start serial_uploader.py on the laptop."
            return rows[0], None
        except Exception as exc:
            return None, f"Could not read the latest sensor value: {exc}"

    @st.fragment(run_every=2)
    def live_height_panel():
        row, error = fetch_latest_height()
        if error:
            st.info(error)
            return
        try:
            height = float(row["object_height"])
            st.metric("OBJECT HEIGHT", f"{height:.1f} cm")
            st.caption(f"Last update (UTC): {row.get('created_at', 'unknown')}")
        except (KeyError, TypeError, ValueError):
            st.warning("The latest database row has an invalid height value.")

    live_height_panel()

    st.markdown(
        "Keep the Arduino connected to the laptop and keep **serial_uploader.py** running. "
        "The hosted website cannot read the laptop's COM port directly; the uploader sends "
        "readings through Supabase."
    )

elif page == "AI Surface Defect Detection":
''') + ai_indented + "\n"

(base / "app.py").write_text(app, encoding="utf-8")

uploader = r'''"""
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

# Set these environment variables before running, or edit the values here.
SERIAL_PORT = os.getenv("ARDUINO_PORT", "COM3")
BAUD_RATE = 9600
SUPABASE_URL = os.getenv("SUPABASE_URL", "").rstrip("/")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
TABLE = "height_readings"

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
            time.sleep(2)  # Arduino may reset when serial opens.
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

                height = max(0.0, float(match.group(1)))
                now = time.monotonic()

                # Limit uploads to at most 2 readings per second.
                if now - last_sent < 0.5:
                    continue

                payload = {
                    "object_height": round(height, 1),
                    "created_at": datetime.now(timezone.utc).isoformat(),
                }
                try:
                    response = requests.post(
                        endpoint,
                        headers=headers,
                        json=payload,
                        timeout=10,
                    )
                    response.raise_for_status()
                    print(f"Uploaded object height: {height:.1f} cm")
                    last_sent = now
                except requests.RequestException as exc:
                    print(f"Upload failed: {exc}")

    except serial.SerialException as exc:
        raise SystemExit(
            f"Could not open {SERIAL_PORT}: {exc}\n"
            "Close Arduino Serial Monitor and verify Tools > Port in Arduino IDE."
        )

if __name__ == "__main__":
    main()
'''
(base / "serial_uploader.py").write_text(uploader, encoding="utf-8")

arduino = r'''/*
  HC-SR04 object height measurement with median filtering + smoothing.
  Sensor-to-ground reference height: 15.0 cm
  TRIG -> D9, ECHO -> D10, VCC -> 5V, GND -> GND

  Serial output stays compatible with serial_uploader.py:
  Distance: 9.0 cm | Object Height: 6.0 cm
*/

const byte TRIG_PIN = 9;
const byte ECHO_PIN = 10;
const float REFERENCE_HEIGHT_CM = 15.0;

const byte SAMPLE_COUNT = 9;
const unsigned long ECHO_TIMEOUT_US = 25000UL;
const float SMOOTH_ALPHA = 0.25;  // lower = steadier but slower response

float smoothedDistance = -1.0;

float readOneDistanceCm() {
  digitalWrite(TRIG_PIN, LOW);
  delayMicroseconds(3);
  digitalWrite(TRIG_PIN, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIG_PIN, LOW);

  unsigned long duration = pulseIn(ECHO_PIN, HIGH, ECHO_TIMEOUT_US);
  if (duration == 0) return -1.0;

  float distance = duration * 0.0343 / 2.0;
  if (distance < 2.0 || distance > 400.0) return -1.0;
  return distance;
}

void sortValues(float values[], byte count) {
  for (byte i = 0; i < count; i++) {
    for (byte j = i + 1; j < count; j++) {
      if (values[j] < values[i]) {
        float temp = values[i];
        values[i] = values[j];
        values[j] = temp;
      }
    }
  }
}

float readMedianDistanceCm() {
  float samples[SAMPLE_COUNT];
  byte validCount = 0;

  for (byte i = 0; i < SAMPLE_COUNT; i++) {
    float value = readOneDistanceCm();
    if (value > 0) {
      samples[validCount++] = value;
    }
    delay(35);  // avoid overlapping ultrasonic echoes
  }

  if (validCount == 0) return -1.0;
  sortValues(samples, validCount);
  return samples[validCount / 2];
}

void setup() {
  pinMode(TRIG_PIN, OUTPUT);
  pinMode(ECHO_PIN, INPUT);
  digitalWrite(TRIG_PIN, LOW);
  Serial.begin(9600);
}

void loop() {
  float medianDistance = readMedianDistanceCm();

  if (medianDistance > 0) {
    if (smoothedDistance < 0) {
      smoothedDistance = medianDistance;
    } else {
      smoothedDistance =
        SMOOTH_ALPHA * medianDistance +
        (1.0 - SMOOTH_ALPHA) * smoothedDistance;
    }

    float objectHeight = REFERENCE_HEIGHT_CM - smoothedDistance;
    if (objectHeight < 0) objectHeight = 0;

    Serial.print("Distance: ");
    Serial.print(smoothedDistance, 1);
    Serial.print(" cm | Object Height: ");
    Serial.print(objectHeight, 1);
    Serial.println(" cm");
  } else {
    Serial.println("No valid echo");
  }

  delay(100);
}
'''
(base / "arduino_height_sensor.ino").write_text(arduino, encoding="utf-8")

requirements = """streamlit>=1.37
numpy
matplotlib
Pillow
torch
anomalib
ultralytics
huggingface_hub
requests
pyserial
"""
(base / "requirements.txt").write_text(requirements, encoding="utf-8")

readme = r'''# Combined Surface Defect Detection + Live Object Height

This project combines:
- YOLO + PatchCore steel-surface defect inspection.
- Live HC-SR04 object-height display.
- A laptop-side serial uploader that forwards Arduino readings to Supabase.
- A Streamlit Cloud app that reads the newest value from Supabase.

## Important architecture note

Streamlit Cloud cannot directly access a USB/COM port on your laptop. Keep `serial_uploader.py` running on the laptop; it reads the Arduino and posts measurements to Supabase. The hosted app polls Supabase every 2 seconds.

## 1. Reduce sensor fluctuations

1. Upload `arduino_height_sensor.ino` to the Uno.
2. Confirm TRIG=D9, ECHO=D10, VCC=5V, GND=GND.
3. Sensor-to-ground distance must be exactly 15 cm for this sketch.
4. Keep the sensor perpendicular to the object. Use a flat, firm test object first.
5. The sketch takes 9 samples, uses their median, and smooths the result. Increase stability by changing `SMOOTH_ALPHA` from `0.25` to `0.15`; response will be slower.
6. If the conveyor object moves quickly, too much smoothing can make height lag behind the object.

## 2. Create the Supabase table

Create a Supabase project, open **SQL Editor**, and run:

```sql
create table public.height_readings (
  id bigint generated by default as identity primary key,
  object_height double precision not null,
  created_at timestamptz not null default now()
);

alter table public.height_readings enable row level security;

create policy "service role can manage height readings"
on public.height_readings
for all
to service_role
using (true)
with check (true);
```

Use the **Project URL** and **service_role/secret key** from Supabase project API settings. Never publish the service-role key in GitHub, source code, screenshots, or a public message. The service-role key bypasses RLS; keep it only in local environment variables and Streamlit Cloud Secrets.

## 3. Configure the Streamlit Cloud app

In the Streamlit Cloud app's **Settings → Secrets**, add:

```toml
SUPABASE_URL = "https://YOUR_PROJECT_ID.supabase.co"
SUPABASE_SERVICE_ROLE_KEY = "YOUR_SUPABASE_SERVICE_ROLE_KEY"
```

Do not commit secrets to GitHub.

## 4. Configure the laptop uploader on Windows

Close Arduino IDE Serial Monitor first. Open Command Prompt in this project folder and install uploader dependencies:

```bat
python -m pip install pyserial requests
```

Set environment variables in that same Command Prompt (replace placeholders):

```bat
set SUPABASE_URL=https://YOUR_PROJECT_ID.supabase.co
set SUPABASE_SERVICE_ROLE_KEY=YOUR_SUPABASE_SERVICE_ROLE_KEY
set ARDUINO_PORT=COM3
python serial_uploader.py
```

Change `COM3` if Arduino IDE → Tools → Port shows another port. Keep this terminal open while you need live data online.

## 5. Deploy the combined app

1. Upload `app.py`, `requirements.txt`, and the other project files to your existing GitHub repository.
2. On Streamlit Community Cloud, set the app's main file to `app.py` (or edit the existing app's main file setting).
3. Add the Supabase secrets described above.
4. Reboot the app after dependencies and secrets are set.
5. Open **Live Object Height** in the sidebar. The displayed value updates approximately every 2 seconds.
6. Choose **AI Surface Defect Detection** for the existing upload-based YOLO and PatchCore workflow.

## Troubleshooting

- **No sensor readings:** check `ARDUINO_PORT`, close Serial Monitor, and check the uploader terminal.
- **Supabase 401/403:** verify the URL/key and that the key is the service-role/secret key.
- **No row appears:** inspect the uploader output for upload errors and verify the table name.
- **Cloud build fails:** AI dependencies such as PyTorch/anomalib are large and version-sensitive. Review Streamlit Cloud logs; pin compatible versions if needed.
- **High fluctuation remains:** secure the sensor, keep the surface perpendicular, avoid measuring edges/angled/soft surfaces, and verify the empty-ground distance is consistently about 15 cm.

This is a demonstration setup. A service-role key has broad database permissions; for production, replace it with a narrowly scoped authenticated endpoint or Edge Function.
'''
(base / "README.md").write_text(readme, encoding="utf-8")

# Basic syntax check on app and uploader.
import ast
ast.parse((base / "app.py").read_text(encoding="utf-8"))
ast.parse((base / "serial_uploader.py").read_text(encoding="utf-8"))

zip_path = Path("/mnt/data/combined_surface_inspection.zip")
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    for p in base.iterdir():
        zf.write(p, arcname=f"combined_surface_inspection/{p.name}")

print("Created project files:")
for p in sorted(base.iterdir()):
    print(f"{p.name}: {p.stat().st_size:,} bytes")
print(f"ZIP: {zip_path} ({zip_path.stat().st_size:,} bytes)")
