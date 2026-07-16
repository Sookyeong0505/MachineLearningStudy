"""
파인튜닝된 GPT-2 모델을 서빙하는 FastAPI 애플리케이션.

실행:
    uv run python run.py
문서:
    http://127.0.0.1:8000/docs
"""

from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import pipeline

# ---------------------------------------------------------------------------
# 모델 경로 / 디바이스 설정
# ---------------------------------------------------------------------------

# main.py 파일 위치를 기준으로 경로를 계산하므로, 어느 디렉터리에서 실행하든 동작합니다.
# app/main.py  ->  parent = app/  ->  parent.parent = 프로젝트 루트
MODEL_PATH = Path(__file__).resolve().parent.parent / "trained_model"


def get_device() -> str:
    """사용 가능한 가속기를 자동으로 선택합니다."""
    if torch.cuda.is_available():
        return "cuda"  # NVIDIA GPU
    if torch.backends.mps.is_available():
        return "mps"  # Apple Silicon (M1/M2/M3)
    return "cpu"


# ---------------------------------------------------------------------------
# FastAPI 앱 초기화
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Text Generation API",
    description="파인튜닝된 GPT-2 모델로 텍스트를 생성하는 API",
    version="1.0.0",
)

# 파이프라인은 모듈 로드 시 딱 한 번만 생성합니다.
# 요청마다 만들면 매번 수백 MB짜리 모델을 디스크에서 다시 읽게 됩니다.
try:
    device = get_device()
    pipe = pipeline("text-generation", model=str(MODEL_PATH), device=device)
    print(f"모델 로드 완료: {MODEL_PATH} (device={device})")
except Exception as e:
    print(f"모델 로드 실패: {e}")
    pipe = None


# ---------------------------------------------------------------------------
# 요청 / 응답 스키마
# ---------------------------------------------------------------------------


class TextGenerationRequest(BaseModel):
    prompt: str = Field(..., description="모델에 입력할 프롬프트")
    max_new_tokens: int = Field(200, ge=1, le=1024, description="생성할 최대 토큰 수")


class TextGenerationResponse(BaseModel):
    generated_text: str


# ---------------------------------------------------------------------------
# 엔드포인트
# ---------------------------------------------------------------------------


@app.get("/")
async def root():
    """환영 메시지."""
    return {"message": "Text Generation API에 오신 것을 환영합니다. 문서는 /docs 를 참고하세요."}


@app.post("/generate", response_model=TextGenerationResponse)
async def generate_text(request: TextGenerationRequest):
    """
    프롬프트를 받아 텍스트를 생성합니다.

    Args:
        request: prompt와 max_new_tokens를 담은 요청 본문

    Returns:
        생성된 텍스트를 담은 TextGenerationResponse
    """
    if pipe is None:
        raise HTTPException(status_code=500, detail="모델이 로드되지 않았습니다")

    try:
        result = pipe(
            request.prompt,
            max_new_tokens=request.max_new_tokens,
            pad_token_id=pipe.tokenizer.eos_token_id,
        )
        generated_text = result[0]["generated_text"]
        return TextGenerationResponse(generated_text=generated_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"텍스트 생성 중 오류: {e}")


@app.get("/health")
async def health_check():
    """API와 모델이 정상 동작하는지 확인합니다."""
    if pipe is None:
        raise HTTPException(status_code=500, detail="모델이 로드되지 않았습니다")
    return {"status": "healthy", "model_loaded": True, "device": device}