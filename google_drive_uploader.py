#!/usr/bin/env python3
"""
google_drive_uploader.py

Uploads a local file (e.g., deliverables zip) to Google Drive using
a FREE personal Google account via OAuth2.

This script uses:
  - credentials.json  (OAuth client credentials from Google Cloud Console)
  - token.json        (saved after first authorization)

Both files live in:
  /home/mykl/webui/filesystem/FiverrMachine/google_credentials/

No Google Workspace is required.

USAGE:
  1) Place credentials.json into the google_credentials/ directory.
  2) Run:
       python3 google_drive_uploader.py /path/to/file.zip "Folder Name In Drive"
  3) On first run, a browser window opens to ask permission.
  4) After that, token.json is reused for silent uploads.
"""

import os
import sys
from pathlib import Path
from typing import Optional

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

BASE = Path("/home/mykl/webui/filesystem/FiverrMachine")
GOOGLE_CREDS_DIR = BASE / "google_credentials"

# Only need access to files we create:
SCOPES = ["https://www.googleapis.com/auth/drive.file"]


def get_credentials() -> Credentials:
    """
    Load or create OAuth credentials.

    Expects:
      - credentials.json in GOOGLE_CREDS_DIR
      - token.json is created after first authorization
    """
    GOOGLE_CREDS_DIR.mkdir(parents=True, exist_ok=True)
    creds = None
    token_path = GOOGLE_CREDS_DIR / "token.json"
    creds_path = GOOGLE_CREDS_DIR / "credentials.json"

    if token_path.exists():
        creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())  # type: ignore[name-defined]
        else:
            if not creds_path.exists():
                raise FileNotFoundError(
                    f"Missing credentials.json in {GOOGLE_CREDS_DIR}.\n"
                    "Visit Google Cloud Console, create an OAuth 'Desktop' client, "
                    "and download credentials.json into that folder."
                )
            flow = InstalledAppFlow.from_client_secrets_file(str(creds_path), SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for future runs
        with token_path.open("w", encoding="utf-8") as token_file:
            token_file.write(creds.to_json())
    return creds


def ensure_folder(service, folder_name: str) -> str:
    """
    Ensure a Drive folder with the given name exists.
    Returns its folder ID.
    """
    query = (
        f"mimeType='application/vnd.google-apps.folder' and "
        f"name='{folder_name}' and trashed=false"
    )
    results = service.files().list(q=query, spaces="drive", fields="files(id, name)").execute()
    items = results.get("files", [])
    if items:
        return items[0]["id"]

    file_metadata = {
        "name": folder_name,
        "mimeType": "application/vnd.google-apps.folder",
    }
    folder = service.files().create(body=file_metadata, fields="id").execute()
    return folder["id"]


def upload_file_to_drive(local_path: Path, folder_name: str) -> Optional[str]:
    """
    Upload the given local file into the specified Drive folder.
    Returns the file's webViewLink if successful.
    """
    creds = get_credentials()
    service = build("drive", "v3", credentials=creds)

    folder_id = ensure_folder(service, folder_name)

    file_metadata = {
        "name": local_path.name,
        "parents": [folder_id],
    }
    media = MediaFileUpload(str(local_path), resumable=True)
    uploaded = service.files().create(
        body=file_metadata, media_body=media, fields="id, webViewLink"
    ).execute()

    file_id = uploaded.get("id")
    link = uploaded.get("webViewLink")
    print(f"[DRIVE] Uploaded {local_path} to Drive file ID {file_id}")
    print(f"[DRIVE] View link: {link}")
    return link


def main():
    if len(sys.argv) != 3:
        print("Usage: python3 google_drive_uploader.py /path/to/file.zip \"Drive Folder Name\"")
        raise SystemExit(1)

    local_path = Path(sys.argv[1]).resolve()
    folder_name = sys.argv[2]

    if not local_path.is_file():
        print(f"[DRIVE] Not a file: {local_path}")
        raise SystemExit(1)

    upload_file_to_drive(local_path, folder_name)


if __name__ == "__main__":
    main()
