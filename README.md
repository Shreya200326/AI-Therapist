# 🐧 PebblePal – Your AI Therapist

PebblePal is a gentle, empathetic AI therapist chatbot powered by Mistral AI and FastAPI. Designed to act like a friendly penguin therapist, Pebble listens to your thoughts and offers comforting, intelligent responses grounded in language model capabilities.

---

## 💡 Project Overview

This project is a mental health companion that processes user messages and responds in a warm, thoughtful manner. The backend is built with FastAPI and connects to Mistral’s `mistral-large-latest` model for generating responses. When no API key is available, it uses mock responses for offline testing.

---

## 🧠 Features

- 🧘 Empathetic chatbot personality (Pebble the Penguin 🐧)
- 💬 Supports natural conversation using Mistral LLM
- 🌐 CORS-enabled API (frontend-ready)
- 🧪 Health check endpoint
- 🛠 Mock response fallback (for development without API)

---

## 🔧 Tech Stack

- **FastAPI** – High-performance Python web framework
- **Mistral AI** – Lightweight, fast, open LLM API
- **HTML/CSS (optional)** – Frontend served via `static/`
- **dotenv** – Environment variable management

