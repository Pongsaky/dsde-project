from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from .auth import decode_access_token, TokenData
from datetime import datetime

security = HTTPBearer()

def get_current_user(credientials: HTTPAuthorizationCredentials = Depends(security)) -> TokenData:
    token = credientials.credentials
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    payload = decode_access_token(token)
    if payload is None:
        raise credentials_exception

    # Check if the token has expired
    if payload.get("exp") < datetime.now().timestamp():
        raise credentials_exception  # Token is expired

    username: str = payload.get("username")
    if username is None:
        raise credentials_exception

    return TokenData(username=username, expireToken=payload.get("exp"))