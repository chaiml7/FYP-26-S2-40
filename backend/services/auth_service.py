import os

import requests
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")
SUPABASE_SECRET_KEY = os.getenv("SUPABASE_SECRET_KEY")


class AuthServiceError(Exception):
    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        self.detail = detail


def _require_config():
    if not SUPABASE_URL or not SUPABASE_ANON_KEY:
        raise AuthServiceError(
            500,
            "SUPABASE_URL and SUPABASE_ANON_KEY must be configured"
        )


def _headers(access_token: str = None):
    headers = {
        "apikey": SUPABASE_ANON_KEY,
        "Content-Type": "application/json",
    }

    if access_token:
        headers["Authorization"] = f"Bearer {access_token}"

    return headers


def _admin_headers():
    if not SUPABASE_URL or not SUPABASE_SECRET_KEY:
        raise AuthServiceError(
            500,
            "SUPABASE_URL and SUPABASE_SECRET_KEY must be configured"
        )

    return {
        "apikey": SUPABASE_SECRET_KEY,
        "Authorization": f"Bearer {SUPABASE_SECRET_KEY}",
        "Content-Type": "application/json",
    }


def _request(method: str, path: str, payload: dict = None, access_token: str = None):
    _require_config()

    response = requests.request(
        method,
        f"{SUPABASE_URL}/auth/v1{path}",
        headers=_headers(access_token),
        json=payload,
        timeout=15,
    )

    if response.status_code >= 400:
        try:
            body = response.json()
        except ValueError:
            body = response.text

        if isinstance(body, dict):
            detail = (
                body.get("msg")
                or body.get("message")
                or body.get("error_description")
                or str(body)
            )
        else:
            detail = str(body)

        raise AuthServiceError(response.status_code, detail)

    if response.status_code == 204 or not response.text:
        return {}

    return response.json()


def _admin_request(method: str, path: str, payload: dict = None):
    response = requests.request(
        method,
        f"{SUPABASE_URL}/auth/v1{path}",
        headers=_admin_headers(),
        json=payload,
        timeout=15,
    )

    if response.status_code >= 400:
        try:
            body = response.json()
        except ValueError:
            body = response.text

        if isinstance(body, dict):
            detail = (
                body.get("msg")
                or body.get("message")
                or body.get("error_description")
                or str(body)
            )
        else:
            detail = str(body)

        raise AuthServiceError(response.status_code, detail)

    if response.status_code == 204 or not response.text:
        return {}

    return response.json()


def create_account(email: str, password: str, username: str = None, full_name: str = None):
    metadata = {}

    if username:
        metadata["username"] = username

    if full_name:
        metadata["full_name"] = full_name

    return _request(
        "POST",
        "/signup",
        {
            "email": email,
            "password": password,
            "data": metadata,
        },
    )


def login(email: str, password: str):
    return _request(
        "POST",
        "/token?grant_type=password",
        {
            "email": email,
            "password": password,
        },
    )


def logout(access_token: str):
    return _request("POST", "/logout", access_token=access_token)


def get_auth_user(access_token: str):
    return _request("GET", "/user", access_token=access_token)


def update_password(access_token: str, new_password: str):
    return _request(
        "PUT",
        "/user",
        {"password": new_password},
        access_token=access_token,
    )


def update_email(access_token: str, email: str):
    return _request(
        "PUT",
        "/user",
        {"email": email},
        access_token=access_token,
    )


def admin_list_users(page: int = 1, per_page: int = 100):
    return _admin_request("GET", f"/admin/users?page={page}&per_page={per_page}")


def admin_get_user(user_id: str):
    return _admin_request("GET", f"/admin/users/{user_id}")
