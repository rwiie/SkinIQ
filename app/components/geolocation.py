"""Detect the user's exact location in the browser (no typing).

Uses the browser Geolocation API (GPS / Wi-Fi), which asks the user for permission once
and needs a secure page (https or localhost). The place name comes from BigDataCloud's
free client-side reverse geocoding, called from the browser as that service intends.
Result, kept in st.session_state[key]["location"]:
  {"status": "ok", "lat", "lon", "accuracy_m", "place"} or {"status": "error", "code", "message"}.
"""

from __future__ import annotations

import streamlit as st

_HTML = """
<div class="geo">
  <span class="msg">Detecting your location…</span>
  <button type="button" class="retry" hidden>Try again</button>
</div>
"""

_CSS = """
.geo { display: flex; gap: 0.75rem; align-items: center; font-family: var(--st-font);
       color: var(--st-text-color); font-size: 0.9rem; min-height: 2rem; }
.geo .msg { opacity: 0.85; }
.geo button { font: inherit; padding: 0.25rem 0.75rem; border-radius: var(--st-base-radius);
              border: 1px solid var(--st-border-color); background: var(--st-secondary-background-color);
              color: var(--st-text-color); cursor: pointer; }
"""

_JS = """
const started = new WeakMap();

function placeName(r) {
  const parts = [r.locality, r.city, r.principalSubdivision, r.countryName]
    .filter((p) => p && String(p).trim());
  return [...new Set(parts)].join(", ");
}

async function reverseGeocode(lat, lon) {
  try {
    const url = "https://api.bigdatacloud.net/data/reverse-geocode-client?latitude=" + lat +
      "&longitude=" + lon + "&localityLanguage=en";
    const ctrl = new AbortController();
    const t = setTimeout(() => ctrl.abort(), 6000);
    const resp = await fetch(url, { signal: ctrl.signal });
    clearTimeout(t);
    if (!resp.ok) return "";
    return placeName(await resp.json());
  } catch (e) {
    return "";
  }
}

const ERRORS = {
  1: "Location access was blocked. Allow location for this site in your browser settings, then press Try again.",
  2: "Your location could not be determined. Check that Location Services are on, then press Try again.",
  3: "Finding your location took too long. Press Try again.",
};

export default function (component) {
  const { data, parentElement, setStateValue } = component;
  const msg = parentElement.querySelector(".msg");
  const retry = parentElement.querySelector(".retry");
  const current = data && data.location;

  const show = (loc) => {
    if (!loc) { msg.textContent = "Detecting your location…"; retry.hidden = true; return; }
    if (loc.status === "ok") {
      msg.textContent = "📍 " + (loc.place || "Location found") + " (accurate to about " +
        Math.round(loc.accuracy_m) + " m)";
      retry.hidden = true;
    } else {
      msg.textContent = loc.message;
      retry.hidden = false;
    }
  };

  const locate = () => {
    show(null);
    if (!window.isSecureContext || !navigator.geolocation) {
      const loc = { status: "error", code: 0, message:
        "Location needs a secure page (https or localhost). Open SkinIQ at http://localhost:8501." };
      show(loc); setStateValue("location", loc); return;
    }
    navigator.geolocation.getCurrentPosition(
      async (pos) => {
        const lat = pos.coords.latitude, lon = pos.coords.longitude;
        const place = await reverseGeocode(lat, lon);
        const loc = { status: "ok", lat, lon, accuracy_m: pos.coords.accuracy, place };
        show(loc); setStateValue("location", loc);
      },
      (err) => {
        const loc = { status: "error", code: err.code, message: ERRORS[err.code] || err.message };
        show(loc); setStateValue("location", loc);
      },
      { enableHighAccuracy: true, timeout: 20000, maximumAge: 10 * 60 * 1000 },
    );
  };

  retry.onclick = locate;
  show(current);
  if (!current && !started.get(parentElement)) {
    started.set(parentElement, true);
    locate();
  }
}
"""

_GEOLOCATION = st.components.v2.component("skiniq_geolocation", html=_HTML, css=_CSS, js=_JS)


def detect_location(key: str = "geo") -> dict | None:
    """Mount the detector; return the latest location result (None while detecting)."""
    current = (st.session_state.get(key) or {}).get("location")
    result = _GEOLOCATION(key=key, data={"location": current}, on_location_change=lambda: None)
    return result.location or current
