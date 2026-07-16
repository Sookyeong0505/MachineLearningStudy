from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline
import os
import sys

# Add the parent directory to sys.path to import modules from the parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Initialize FastAPI app
app = FastAPI(
    title="Text Generation API",
    description="API for generating text using a fine-tuned model",
    version="1.0.0"
)

# Initialize the model pipeline
try:
    pipe = pipeline("text-generation", model="../trained_model", device="mps")
except Exception as e:
    # Fallback to CPU if MPS is not available
    try:
        pipe = pipeline("text-generation", model="../trained_model", device="cpu")
    except Exception as e:
        print(f"Error loading model: {e}")
        pipe = None


# Request model
class TextGenerationRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 200


# Response model
class TextGenerationResponse(BaseModel):
    generated_text: str


@app.get("/")
async def root():
    return {"message": "Welcome to the Text Generation API"}


@app.post("/generate", response_model=TextGenerationResponse)
async def generate_text(request: TextGenerationRequest):
    """
    Generate text based on the provided prompt.

    Args:
        request: TextGenerationRequest containing the prompt and generation parameters

    Returns:
        TextGenerationResponse with the generated text
    """
    if pipe is None:
        raise HTTPException(status_code=500, detail="Model not loaded properly")

    try:
        result = pipe(
            request.prompt,
            max_new_tokens=request.max_new_tokens,
            pad_token_id=pipe.tokenizer.eos_token_id
        )

        # Extract the generated text from the result
        generated_text = result[0]['generated_text']

        return TextGenerationResponse(generated_text=generated_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating text: {str(e)}")


# Health check endpoint
@app.get("/health")
async def health_check():
    """Check if the API and model are working properly."""
    if pipe is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    return {"status": "healthy", "model_loaded": True}