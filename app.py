"""
Builds the combined "Surface Inspection + Live Object Height" project.

Usage:
    python build_combined_project.py                       # looks for the source file automatically
    python build_combined_project.py path/to/original.py   # or pass it explicitly

Output (in the current folder):
    combined_surface_inspection/            app.py, serial_uploader.py,
                                            arduino_height_sensor.ino,
                                            requirements.txt, README.md
    combined_surface_inspection.zip
"""
import ast
import sys
import textwrap
import zipfile
from pathlib import Path

# ------------------------------------------------------------------
# 0. Locations (no hard-coded /mnt/data or /tmp paths)
# ------------------------------------------------------------------
OUT_DIR = Path.cwd() / "combined_surface_inspection"
ZIP_PATH = Path.cwd() / "combined_surface_inspection.zip"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def find_source() -> Path:
    """Find the original single-page AI app source."""
    if len(sys.argv) > 1:
        return Path(sys.argv[1])
    candidates = [
        Path.cwd() / "Pasted text(2).txt",
        Path.cwd() / "original_app.py",
        Path.cwd() / "app_original.py",
        Path("/mnt/data/Pasted text(2).txt"),
    ]
    for c in candidates:
        if c.exists():
            return c
    raise SystemExit(
        "Original source file not found. Put it in this folder as "
        "'original_app.py' or run:\n"
        "    python build_combined_project.py path/to/your_original_app.py"
    )


def find_marker(text: str, marker: str) -> int:
    """str.index with a readable error message."""
    pos = text.find(marker)
    if pos == -1:
        raise SystemExit(
            "Section marker not found in the original source:\n"
            f"{marker!r}\n"
            "Make sure the original file still has its '# ====' section headers."
        )
    return pos


SEP = "# ============================================================\n"
source = find_source().read_text(encoding="utf-8")

# ------------------------------------------------------------------
# 1. Split the original file into pieces
#       head     : imports (section 1)
#       (removed): original page config + title (section 2)
#       helpers  : HF token, checkpoints, model loading ... (sections 3-7)
#       ai_body  : AI page UI (section 8 onward)
# ------------------------------------------------------------------
page_start = find_marker(source, SEP + "# 2. STREAMLIT PAGE CONFIGURATION")
hf_start = find_marker(source, SEP + "# 3. HUGGING FACE TOKEN")
ai_start = find_marker(source, SEP + "# 8. MODEL INFORMATION")

head = source[:page_start]
helpers = source[hf_start:ai_start]
ai_body = source[ai_start:]

# Extra imports go at the very top (after any __future__ import).
extra_imports = "import requests\n"
if head.lstrip().startswith("from __future__"):
    first_nl = head.index("\n") + 1
    head = head[:first_nl] + extra_imports + head[first_nl:]
else:
    head = extra_imports + head

# Indent the AI section so it only runs when that page is selected.
# (textwrap.indent skips blank lines automatically.)
ai_indented = textwrap.indent(ai_body, "    ")

# ------------------------------------------------------------------
# 2. Templates
# ------------------------------------------------------------------

# FIX: set_page_config MUST be the first Streamlit call. In the original build
# it was placed AFTER the helpers (which already call st.*), which raises
# StreamlitAPIException. It now sits directly after the imports.
PAGE_CONFIG = '''
# ============================================================
# PAGE CONFIGURATION (must be the first Streamlit command)
# ============================================================
st.set_page_config(
    page_title="Surface Inspection & Object Height",
    page_icon="🔍",
    layout="wide",
)

'''

COMBINED_NAV = '''

# ============================================================
# LIVE OBJECT HEIGHT HELPERS
# ============================================================
def _secret(*names, default=""):
    """Read the first available Streamlit secret; never crash if secrets.toml is missing."""
    for name in names:
        try:
            value = st.secrets[name]
            if value:
                return str(value)
        except Exception:
            continue
    return default


@st.cache_data(ttl=2, show_spinner=False)
def fetch_latest_height():
    """Read the newest sensor reading from the Supabase REST API."""
    url = _secret("SUPABASE_URL").rstrip("/")
    # Prefer the read-only anon key; fall back to the service-role key.
    key = _secret("SUPABASE_ANON_KEY", "SUPABASE_SERVICE_ROLE_KEY")
    if not url or not key:
        return None, "Add SUPABASE_URL and SUPABASE_ANON_KEY in Streamlit Secrets."

    endpoint = f"{url}/rest/v1/height_readings"
    headers = {"apikey": key, "Authorization": f"Bearer {key}"}
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
            return None, "No sensor readings yet. Start serial_uploader.py on the laptop."
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


# ============================================================
# COMBINED APP NAVIGATION
# ============================================================
st.title("🔍 Surface Inspection & Object Height")
st.caption("AI steel-surface inspection plus live Arduino HC-SR04 height measurement.")

page = st.sidebar.radio(
    "Choose project module",
    ["Live Object Height", "AI Surface Defect Detection"],
)

if page == "Live Object Height":
    st.header("Live Object Height")
    st.caption("The laptop-side serial bridge sends filtered sensor readings to the cloud database.")
    live_height_panel()
    st.markdown(
        "Keep the Arduino connected to the laptop and keep **serial_uploader.py** running. "
        "The hosted website cannot read the laptop's COM port directly; the uploader sends "
        "readings through Supabase."
    )

elif page == "AI Surface Defect Detection":
'''

UPLOADER = r'''"""
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
                last_sent = now  # set before the request so failures don't cause a retry storm

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
'''

ARDUINO = r'''/*
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

REQUIREMENTS = """streamlit>=1.37
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

README = r'''# Combined Surface Defect Detection + Live Object Height

- YOLO + PatchCore steel-surface defect inspection.
- Live HC-SR04 object-height display.
- A laptop-side serial uploader that forwards Arduino readings to Supabase.
- A Streamlit Cloud app that reads the newest value from Supabase.

## Architecture note

Streamlit Cloud cannot access a USB/COM port on your laptop. Keep `serial_uploader.py`
running on the laptop; it reads the Arduino and posts measurements to Supabase.
The hosted app polls Supabase every 2 seconds.

## 1. Arduino

1. Upload `arduino_height_sensor.ino` to the Uno.
2. Wiring: TRIG=D9, ECHO=D10, VCC=5V, GND=GND.
3. Sensor-to-ground distance must be exactly 15 cm (`REFERENCE_HEIGHT_CM`).
4. Keep the sensor perpendicular to a flat, firm object.
5. For steadier readings lower `SMOOTH_ALPHA` (0.25 -> 0.15); response becomes slower.

## 2. Supabase table

Run in **SQL Editor**:

```sql
create table public.height_readings (
  id bigint generated by default as identity primary key,
  object_height double precision not null,
  created_at timestamptz not null default now()
);

alter table public.height_readings enable row level security;

-- The website only needs to READ the newest value (anon key).
create policy "anon can read height readings"
on public.height_readings
for select
to anon
using (true);

-- Writes come from the laptop uploader using the service-role key (bypasses RLS).
```

Never put the service-role key in GitHub, the Streamlit app, screenshots or messages.

## 3. Streamlit Cloud secrets (Settings -> Secrets)

```toml
SUPABASE_URL = "https://YOUR_PROJECT_ID.supabase.co"
SUPABASE_ANON_KEY = "YOUR_SUPABASE_ANON_PUBLIC_KEY"
```

## 4. Laptop uploader (Windows)

Close Arduino Serial Monitor first, then:

```bat
python -m pip install pyserial requests
set SUPABASE_URL=https://YOUR_PROJECT_ID.supabase.co
set SUPABASE_SERVICE_ROLE_KEY=YOUR_SUPABASE_SERVICE_ROLE_KEY
set ARDUINO_PORT=COM3
python serial_uploader.py
```

## 5. Deploy

1. Push `app.py`, `requirements.txt` and the other files to GitHub.
2. In Streamlit Community Cloud set the main file to `app.py`.
3. Add the secrets and reboot the app.

## Troubleshooting

- **No sensor readings:** check `ARDUINO_PORT`, close Serial Monitor, read the uploader output.
- **Supabase 401/403:** check URL/key and that the SELECT policy for `anon` exists.
- **Cloud build fails:** PyTorch/anomalib are large and version-sensitive; pin versions from the logs.
- **Height page blocked by the AI section:** if the original code calls `st.stop()` at module level
  (e.g. missing Hugging Face token), it stops the whole app, including the height page.
'''

# ------------------------------------------------------------------
# 3. Assemble app.py
# ------------------------------------------------------------------
app = head + PAGE_CONFIG + helpers + COMBINED_NAV + ai_indented + "\n"

files = {
    "app.py": app,
    "serial_uploader.py": UPLOADER,
    "arduino_height_sensor.ino": ARDUINO,
    "requirements.txt": REQUIREMENTS,
    "README.md": README,
}
for name, content in files.items():
    (OUT_DIR / name).write_text(content, encoding="utf-8")

# ------------------------------------------------------------------
# 4. Syntax check (fails loudly instead of shipping a broken app)
# ------------------------------------------------------------------
for name in ("app.py", "serial_uploader.py"):
    try:
        ast.parse((OUT_DIR / name).read_text(encoding="utf-8"), filename=name)
    except SyntaxError as exc:
        raise SystemExit(f"Syntax error in generated {name}: {exc}")

# ------------------------------------------------------------------
# 5. Zip
# ------------------------------------------------------------------
with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as zf:
    for p in sorted(OUT_DIR.iterdir()):
        zf.write(p, arcname=f"combined_surface_inspection/{p.name}")

print("Created project files:")
for p in sorted(OUT_DIR.iterdir()):
    print(f"  {p.name}: {p.stat().st_size:,} bytes")
print(f"ZIP: {ZIP_PATH} ({ZIP_PATH.stat().st_size:,} bytes)")
