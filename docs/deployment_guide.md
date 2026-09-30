# FaceVital AI — Production Deployment Guide

This guide details how to deploy FaceVital AI so you can share a live, working URL with interviewers and evaluators.

---

## Architecture for Production

Because FaceVital AI has two components:
1. **Frontend (React + Vite + TypeScript):** Best deployed on **Vercel** (Global CDN, free HTTPS required for webcam permissions).
2. **Backend (FastAPI + PyTorch + rPPG):** Best deployed on **Render** or **Railway** (free Python runtime with memory for PyTorch).

```
   [User's Browser (Webcam)]
             │
             ▼ HTTPS (Webcam access requires HTTPS!)
┌─────────────────────────┐
│   Vercel (Frontend)     │  ---> https://facevital.vercel.app
└────────────┬────────────┘
             │ REST API calls
             ▼
┌─────────────────────────┐
│   Render (FastAPI API)  │  ---> https://facevital-api.onrender.com
└─────────────────────────┘
```

---

## Option 1: Vercel (Frontend) + Render (Backend) [Recommended]

### Step 1: Deploy Backend to Render (5 minutes)
1. Push your code to GitHub.
2. Sign up / Log in to [Render.com](https://render.com) (free).
3. Click **New +** -> **Web Service**.
4. Connect your GitHub repository.
5. Set:
   - **Name:** `facevital-backend`
   - **Root Directory:** (leave empty or set to root)
   - **Runtime:** `Python 3`
   - **Build Command:** `pip install -r backend/requirements.txt`
   - **Start Command:** `uvicorn backend.app.main:app --host 0.0.0.0 --port $PORT`
6. Add Environment Variable:
   - `CORS_ORIGINS`: `*` (or your Vercel URL)
   - `DATABASE_URL`: `sqlite+aiosqlite:///facevital.db`
7. Click **Deploy Web Service**.
8. Copy your backend URL (e.g. `https://facevital-backend.onrender.com`).

---

### Step 2: Deploy Frontend to Vercel (3 minutes)
1. Sign up / Log in to [Vercel.com](https://vercel.com).
2. Click **Add New...** -> **Project**.
3. Import your GitHub repository.
4. Set:
   - **Root Directory:** Select `frontend`
   - **Framework Preset:** `Vite`
5. Under **Environment Variables**, add:
   - **Name:** `VITE_API_BASE_URL`
   - **Value:** `https://facevital-backend.onrender.com` (your Render URL from Step 1)
6. Click **Deploy**.
7. Done! You will get an instant live HTTPS link like `https://facevital-ai.vercel.app`.

---

## Option 2: Deploy Frontend on Vercel Standalone (Immediate)

If you need a shareable link **right now** without setting up a backend cloud server:
- Deploy the `frontend/` directory to Vercel.
- The app already has a **built-in client-side simulation demo mode**!
- Interviewers can immediately open the link, test the UI, run the simulation, see the real-time BVP waveform and vitals, and export JSON reports.

---

## Why HTTPS is Mandatory
Modern browsers (Chrome, Safari, Firefox, Edge) block webcam access (`navigator.mediaDevices.getUserMedia`) on unencrypted HTTP connections.
Both **Vercel** and **Render** provide automatic free SSL certificates (`https://`), ensuring the interviewer's browser will allow webcam access without security warnings.
