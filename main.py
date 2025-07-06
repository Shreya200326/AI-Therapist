from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
from dotenv import load_dotenv
import logging

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="PebblePal AI Therapist Backend",
    description="Backend for PebblePal AI therapist chatbot using FastAPI and Mistral AI.",
    version="1.0.0"
)

origins = [
    "http://localhost",
    "http://localhost:8000",
    "http://127.0.0.1",
    "http://127.0.0.1:8000",
  
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    app.mount("/static", StaticFiles(directory="static"), name="static")
except RuntimeError:
    logger.warning("Static directory not found. Create a 'static' directory with your frontend files.")


mistral_api_key = os.getenv("MISTRAL_API_KEY")
client = MistralClient(api_key=mistral_api_key) if mistral_api_key else None

PEBBLE_SYSTEM_PROMPT = """
You are Pebble, a cute, empathetic, and gentle penguin therapist...
(keep rest of the prompt the same)
"""

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str
    status: str = "success"

MOCK_RESPONSES = [
    "It sounds like you're carrying a lot right now, like a little iceberg of worries...",
    "I hear you, and what you're feeling is completely valid...",
]
mock_response_counter = 0

@app.get("/", response_class=HTMLResponse)
async def root():
    try:
        return FileResponse("static/index.html")
    except FileNotFoundError:
        return HTMLResponse("<h1>PebblePal Backend Running!</h1><p>But no frontend found.</p>")

@app.get("/health")
async def health():
    return {"status": "healthy", "mistral_ready": bool(client)}

@app.post("/chat", response_model=ChatResponse)
async def chat_with_pebble(request: ChatRequest):
    message = request.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    try:
        if client:
            messages = [
                ChatMessage(role="system", content=PEBBLE_SYSTEM_PROMPT),
                ChatMessage(role="user", content=message)
            ]
            response = client.chat(
                model="mistral-large-latest",
                messages=messages,
                max_tokens=300,
                temperature=0.7
            )
            answer = response.choices[0].message.content
        else:
            global mock_response_counter
            answer = MOCK_RESPONSES[mock_response_counter % len(MOCK_RESPONSES)]
            mock_response_counter += 1

        return ChatResponse(response=answer)
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail="Pebble had a hiccup. Try again later.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
