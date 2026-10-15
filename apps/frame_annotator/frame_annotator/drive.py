"""Google Drive v3 client: per-thread services, retry with backoff, create-or-update.

httplib2 is not thread-safe, so every thread gets its own service object (built
once, then reused). Credentials come from the environment:

* ``GOOGLE_AUTHORIZED_USER_JSON`` / ``GOOGLE_AUTHORIZED_USER_FILE`` - a user
  OAuth token (needed for a personal Gmail Drive: service accounts have no
  storage quota there and cannot create files);
* ``GOOGLE_SERVICE_ACCOUNT_JSON`` - a service account (works with a Shared Drive).
"""
from __future__ import annotations

import io
import json
import logging
import random
import threading
import time
from typing import Mapping, Optional

log = logging.getLogger(__name__)

SCOPES = ["https://www.googleapis.com/auth/drive"]
FOLDER_MIME = "application/vnd.google-apps.folder"
MAX_ATTEMPTS = 6
_RETRY_STATUSES = {429, 500, 502, 503, 504}
_RETRY_REASONS = {"rateLimitExceeded", "userRateLimitExceeded", "backendError"}


class DriveQuotaError(RuntimeError):
    """Raised when Drive refuses to create a file because the owner has no quota."""


def credentials_from_env(env: Mapping):
    raw = env.get("GOOGLE_SERVICE_ACCOUNT_JSON")
    if raw:
        from google.oauth2 import service_account

        return service_account.Credentials.from_service_account_info(json.loads(raw), scopes=SCOPES)
    from google.oauth2.credentials import Credentials

    raw = env.get("GOOGLE_AUTHORIZED_USER_JSON")
    if raw:
        return Credentials.from_authorized_user_info(json.loads(raw))
    path = env.get("GOOGLE_AUTHORIZED_USER_FILE")
    if path:
        return Credentials.from_authorized_user_file(path)
    raise RuntimeError(
        "no Google credentials: set GOOGLE_AUTHORIZED_USER_JSON, GOOGLE_AUTHORIZED_USER_FILE "
        "or GOOGLE_SERVICE_ACCOUNT_JSON"
    )


def _escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace("'", "\\'")


class DriveClient:
    def __init__(self, credentials):
        self._credentials = credentials
        self._local = threading.local()

    def _service(self):
        service = getattr(self._local, "service", None)
        if service is None:
            from googleapiclient.discovery import build

            service = build("drive", "v3", credentials=self._credentials, cache_discovery=False)
            self._local.service = service
        return service

    def _call(self, make_request):
        """Run ``make_request(service).execute()``, retrying transient failures."""
        from googleapiclient.errors import HttpError

        for attempt in range(MAX_ATTEMPTS):
            try:
                return make_request(self._service()).execute()
            except HttpError as error:
                status = error.resp.status
                reason = _reason(error)
                if reason == "storageQuotaExceeded":
                    raise DriveQuotaError(
                        "Drive refused the upload: the account that owns the credentials has no "
                        "storage quota. Use a user OAuth token or a Shared Drive."
                    ) from error
                transient = status in _RETRY_STATUSES or reason in _RETRY_REASONS
                if not transient or attempt == MAX_ATTEMPTS - 1:
                    raise
                delay = min(60.0, 2 ** attempt) + random.random()
                log.warning("drive %s (%s); retry %d in %.1fs", status, reason, attempt + 1, delay)
                time.sleep(delay)

    # -- queries -----------------------------------------------------------
    def _find(self, name: str, parent_id: str, folder_only: bool = False) -> Optional[str]:
        query = f"name = '{_escape(name)}' and '{_escape(parent_id)}' in parents and trashed = false"
        if folder_only:
            query += f" and mimeType = '{FOLDER_MIME}'"
        response = self._call(lambda s: s.files().list(
            q=query, fields="files(id)", pageSize=2,
            supportsAllDrives=True, includeItemsFromAllDrives=True,
        ))
        files = response.get("files", [])
        return files[0]["id"] if files else None

    def find(self, name: str, parent_id: str) -> Optional[str]:
        return self._find(name, parent_id)

    def ensure_folder(self, name: str, parent_id: str) -> str:
        existing = self._find(name, parent_id, folder_only=True)
        if existing:
            return existing
        body = {"name": name, "mimeType": FOLDER_MIME, "parents": [parent_id]}
        return self._call(lambda s: s.files().create(
            body=body, fields="id", supportsAllDrives=True))["id"]

    def list_names(self, parent_id: str) -> dict:
        names, token = {}, None
        while True:
            response = self._call(lambda s: s.files().list(
                q=f"'{_escape(parent_id)}' in parents and trashed = false",
                fields="nextPageToken, files(id, name)", pageSize=1000, pageToken=token,
                supportsAllDrives=True, includeItemsFromAllDrives=True,
            ))
            names.update({f["name"]: f["id"] for f in response.get("files", [])})
            token = response.get("nextPageToken")
            if not token:
                return names

    # -- content -----------------------------------------------------------
    def upload(self, name: str, data: bytes, parent_id: str,
               mime: str = "application/octet-stream", file_id: Optional[str] = None) -> str:
        """Create ``name`` under ``parent_id``, or replace its content if it exists."""
        from googleapiclient.http import MediaIoBaseUpload

        file_id = file_id or self._find(name, parent_id)

        def request(service):
            media = MediaIoBaseUpload(io.BytesIO(data), mimetype=mime, resumable=False)
            if file_id:
                return service.files().update(
                    fileId=file_id, media_body=media, fields="id", supportsAllDrives=True)
            return service.files().create(
                body={"name": name, "parents": [parent_id]}, media_body=media,
                fields="id", supportsAllDrives=True)

        return self._call(request)["id"]

    def download(self, file_id: str) -> bytes:
        return self._call(lambda s: s.files().get_media(fileId=file_id, supportsAllDrives=True))


def _reason(error) -> str:
    try:
        return json.loads(error.content.decode("utf-8"))["error"]["errors"][0]["reason"]
    except (ValueError, KeyError, IndexError, AttributeError, TypeError):
        return ""
