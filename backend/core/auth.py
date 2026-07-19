"""
auth.py — JWT-based auth.

Endpoints:
  POST /auth/login   → returns access_token
  GET  /auth/me      → returns current user profile

FastAPI dependency:
  get_current_user(token) → user dict or 401
"""

from __future__ import annotations
from datetime import datetime, timedelta, timezone
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel

import backend.services.database as db
from backend.core.config import get_settings
from backend.core.request_context import current_access_token

router = APIRouter(prefix="/auth", tags=["auth"])
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: dict


class UserProfile(BaseModel):
    id: str
    username: str
    name: str
    owned_products: list[dict]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


def hash_password(plain: str) -> str:
    return pwd_context.hash(plain)


def create_access_token(data: dict) -> str:
    s = get_settings()
    payload = data.copy()
    payload["exp"] = datetime.now(timezone.utc) + timedelta(minutes=s.jwt_expire_minutes)
    # Increment 10: include the claims Postgres RLS expects (role/aud) and sign
    # with the Supabase JWT secret when configured, so a per-request user-JWT
    # client is accepted by RLS. Falls back to the app's own jwt_secret in legacy
    # mode; the extra claims are harmless there. `sub` is public.users.id.
    payload.setdefault("role", "authenticated")
    payload.setdefault("aud", "authenticated")
    return jwt.encode(payload, s.auth_signing_secret, algorithm=s.jwt_algorithm)


async def get_current_user(token: Annotated[str, Depends(oauth2_scheme)]) -> dict:
    s = get_settings()
    credentials_exc = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        # verify_aud False: we only need `sub`; Supabase RLS validates the token
        # itself. This also decodes both legacy tokens (no aud) and Increment 10
        # tokens (aud=authenticated) with the same call.
        payload = jwt.decode(
            token, s.auth_signing_secret,
            algorithms=[s.jwt_algorithm], options={"verify_aud": False},
        )
        user_id: str | None = payload.get("sub")
        if user_id is None:
            raise credentials_exc
    except JWTError:
        raise credentials_exc

    user = db.get_user_by_id(user_id)
    if user is None:
        raise credentials_exc
    # Increment 10: expose the raw token so user-owned DB queries can run through
    # a per-request user-JWT client (RLS enforcement).
    current_access_token.set(token)
    return user


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.post("/login", response_model=Token)
async def login(form_data: Annotated[OAuth2PasswordRequestForm, Depends()]):
    user = db.get_user_by_username(form_data.username)
    if not user or not verify_password(form_data.password, user["password_hash"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
        )
    token = create_access_token({"sub": user["id"]})
    safe_user = {k: v for k, v in user.items() if k != "password_hash"}
    return Token(access_token=token, user=safe_user)


@router.get("/me", response_model=UserProfile)
async def me(current_user: Annotated[dict, Depends(get_current_user)]):
    return UserProfile(**current_user)
