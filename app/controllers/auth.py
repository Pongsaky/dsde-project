from fastapi import Form, HTTPException, status, Depends
from app.auth.auth import TokenData, verify_code, create_access_token
from app.auth.dependencies import get_current_user
from fastapi.responses import JSONResponse

def login(code: str = Form(...)):
    if not verify_code(code):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid code",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token = create_access_token(
        data={"username": "afastadmin"}
    )
    return {"access_token": access_token, "token_type": "bearer"}

def get_me(current_user: TokenData = Depends(get_current_user)):
    if current_user.username == "afastadmin":
        return JSONResponse(status_code=status.HTTP_200_OK, content={"username": current_user.username, "expireToken": current_user.expireToken})
    else:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid user",
            headers={"WWW-Authenticate": "Bearer"},
        )