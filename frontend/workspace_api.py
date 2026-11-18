import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import streamlit as st

BACKEND = os.environ.get("BACKEND_URL", "http://localhost:8000").rstrip("/")
TOKEN_FILE = Path(os.environ.get("WORKSPACE_API_TOKEN_FILE", "/run/secrets/workspace_api_token"))


class ApiError(Exception):
    def __init__(self, message, status=None):
        super().__init__(message)
        self.status = status


@st.cache_resource
def connection():
    return httpx.Client(base_url=BACKEND, timeout=httpx.Timeout(20, connect=5), trust_env=False)


def api(method, path, *, timeout=20, raw=False, client=None, **kwargs):
    try:
        token = TOKEN_FILE.read_text(encoding="utf-8").rstrip("\r\n")
    except OSError:
        raise ApiError("The frontend cannot read its workspace token. Check SECRET_KEY and the Compose secret.") from None
    if not token:
        raise ApiError("Configure SECRET_KEY and recreate the frontend service.")
    try:
        response = (client or connection()).request(method, f"/api/v1{path}", headers={"Authorization": f"Bearer {token}"},
                                        timeout=httpx.Timeout(timeout, connect=5), **kwargs)
    except httpx.TimeoutException:
        raise ApiError("The backend is taking too long to respond. Check service health before retrying.") from None
    except httpx.HTTPError:
        raise ApiError("The frontend could not reach the backend.") from None
    if response.is_error:
        try:
            payload = response.json()
            detail = payload.get("detail", "Backend error.") if isinstance(payload, dict) else "Backend error."
        except ValueError:
            detail = "The backend returned an unexpected response."
        if isinstance(detail, list):
            detail = "\n".join(
                f"{item.get('filename') or item.get('field', 'Input')} · {item.get('error') or item.get('message', '')}"
                for item in detail if isinstance(item, dict)
            ) or "Check the request values."
        raise ApiError(detail, response.status_code)
    if raw:
        return response.content
    return response.json() if response.content else None


def pages(path, **filters):
    offset = 0
    while True:
        rows = api("GET", path, params={**filters, "limit": 100, "offset": offset})
        yield from rows
        if len(rows) < 100:
            return
        offset += len(rows)


@st.cache_data(ttl=8, show_spinner=False)
def health():
    def probe(path):
        try:
            return connection().get(path, timeout=2).status_code == 200
        except httpx.HTTPError:
            return None

    live = probe("/health/live")
    return live, probe("/health/ready") if live else None


@st.cache_data(ttl=45, show_spinner=False)
def diagnostics(check_external=False):
    return api("GET", "/settings", params={"check_external": str(check_external).lower()}, timeout=35)


def _question_count(identifier, client):
    before = None
    total = 0
    while True:
        params = {"limit": 100}
        if before is not None:
            params["before"] = before
        try:
            history = api("GET", f"/conversations/{identifier}", params=params, client=client)
        except ApiError as exc:
            if exc.status == 404:
                return 0
            raise
        messages = history["messages"]
        total += sum(message["role"] == "user" for message in messages)
        if not history["has_older"] or not messages:
            return total
        before = messages[0]["id"]


@st.cache_data(ttl=30, show_spinner=False)
def saved_question_count(identifiers):
    # Use the existing history API for this personal workspace and bound concurrent reads.
    client = connection()
    with ThreadPoolExecutor(max_workers=6) as executor:
        return sum(executor.map(lambda identifier: _question_count(identifier, client), identifiers))
