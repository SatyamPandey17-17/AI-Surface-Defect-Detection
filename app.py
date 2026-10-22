from pathlib import Path

import requests
import streamlit as st

# ============================================================
# PAGE CONFIGURATION (must be the first Streamlit command)
# ============================================================
st.set_page_config(
    page_title="Surface Inspection & Object Height",
    page_icon="🔍",
    layout="wide",
)


# ============================================================
# LIVE OBJECT HEIGHT HELPERS
# ============================================================
def _secret(*names, default=""):
    """Read the first available Streamlit secret; never crash if secrets are missing."""
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


def height_page():
    st.title("🔍 Surface Inspection & Object Height")
    st.header("Live Object Height")
    st.caption("The laptop-side serial bridge sends filtered sensor readings to the cloud database.")
    live_height_panel()
    st.markdown(
        "Keep the Arduino connected to the laptop and keep **serial_uploader.py** running. "
        "The hosted website cannot read the laptop's COM port directly; the uploader sends "
        "readings through Supabase."
    )


# Streamlit only executes the selected page, so the heavy AI script
# (torch, anomalib, ultralytics) runs ONLY when the AI page is opened.
def ai_missing_page():
    st.title("🧠 AI Surface Defect Detection")
    st.warning(
        "ai_page.py was not found in the GitHub repository. Upload it next to "
        "app.py (same folder, exact lowercase name) and reboot the app."
    )


pages = [st.Page(height_page, title="Live Object Height", icon="📏", default=True)]

if (Path(__file__).parent / "ai_page.py").exists():
    pages.append(st.Page("ai_page.py", title="AI Surface Defect Detection", icon="🧠"))
else:
    pages.append(st.Page(ai_missing_page, title="AI Surface Defect Detection", icon="🧠"))

navigation = st.navigation(pages)
navigation.run()
