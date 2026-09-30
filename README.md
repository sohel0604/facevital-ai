# FaceVital AI — Web-Based Non-Contact Facial Health Monitoring System

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![React 19](https://img.shields.io/badge/react-19-cyan.svg)](https://react.dev/)
[![TypeScript](https://img.shields.io/badge/typescript-5.9-blue.svg)](https://www.typescriptlang.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-green.svg)](https://fastapi.tiangolo.com/)
[![Tests](https://img.shields.io/badge/tests-34%20passed-brightgreen.svg)]()

> ⚠️ **RESEARCH PROTOTYPE NOTICE**  
> FaceVital AI is an experimental computer vision and machine learning platform designed for academic benchmarking and algorithmic research in remote photoplethysmography (rPPG). It is **NOT** a medical device and must **NOT** be used for clinical diagnosis, treatment planning, or life-critical monitoring.

---

## 🚀 Key Capabilities

- **Remote Photoplethysmography (rPPG):**
  - **POS (Plane-Orthogonal-to-Skin):** State-of-the-art optical pulse extraction robust to subtle head motion and skin tone differences.
  - **CHROM (Chrominance-based):** Ultra-fast lightweight rPPG algorithm ($< 25\text{ ms}$ extraction time).
  - **Bandpass Butterworth Filtering:** 2nd-order zero-phase filtering across the physiological cardiac range ($0.7\text{--}3.0\text{ Hz} \equiv 42\text{--}180\text{ bpm}$).
  - **Welch Spectral Peak Detection:** Accurate heart rate estimation with confidence metrics.
- **Real-Time Signal Quality Index (SQI):**
  - Composite multi-factor scoring: Illumination, head motion, inter-ROI consistency, SNR, and periodicity.
  - Automatic quality gates preventing inference on corrupted or insufficient video signals.
- **Multi-Task Biomarker Neural Network:**
  - 1D Temporal Convolutional Neural Network with dilated residual blocks and multi-scale spatial pyramid pooling.
  - Multi-task heads for Systolic BP, Diastolic BP, Blood Glucose, and Total Cholesterol estimation.
  - Fully decoupled architecture supporting calibrated checkpoint weights or transparent research placeholder modes.
- **Premium Dark-Mode Web Dashboard:**
  - Real-time webcam video feed with facial bounding box and 3 anatomical ROI overlays (Forehead, Left Cheek, Right Cheek).
  - Dynamic live Blood Volume Pulse (BVP) chart using Recharts.
  - Interactive "Run Simulation" demo mode for instant verification without physical camera hardware.
  - Session reporting with downloadable JSON physiological assessment reports.
- **Production-Grade Backend Architecture:**
  - High-performance asynchronous FastAPI server with Pydantic v2 schemas.
  - Structured request logging, rate limiting, and CORS middleware.
  - PostgreSQL / SQLite database models for sessions, audit logs, and measurements.

---

## 📂 Project Structure

```
facevital-ai/
├── backend/
│   ├── app/
│   │   ├── api/v1/endpoints.py      # REST API route handlers
│   │   ├── core/config.py           # Pydantic settings & environment configuration
│   │   ├── db/database.py           # Async database sessions & SQLite/Postgres engine
│   │   ├── models/database_models.py# SQLAlchemy ORM models (Session, Measurement)
│   │   ├── schemas/api_schemas.py   # Pydantic request/response schemas
│   │   └── services/inference_service.py # Core inference & rPPG orchestration
│   ├── ml/
│   │   └── biomarker_model.py       # MultiTaskBiomarkerModel (1D-CNN)
│   ├── rppg/
│   │   ├── pos.py                   # Plane-Orthogonal-to-Skin algorithm
│   │   ├── chrom.py                 # Chrominance-based rPPG method
│   │   ├── heart_rate.py            # Welch PSD & peak detection HR estimation
│   │   ├── signal_quality.py        # Optical Signal Quality Index (SQI)
│   │   ├── filters.py               # Butterworth bandpass & normalization
│   │   └── roi.py                   # Facial ROI parsing & spatial combining
│   └── requirements.txt             # Python dependencies
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── VitalCard.tsx        # Vital sign display card with confidence bar
│   │   │   ├── BVPChart.tsx         # Real-time BVP waveform chart (Recharts)
│   │   │   ├── SignalQualityBar.tsx # Optical SQI status & breakdown
│   │   │   └── ResearchDisclaimer.tsx # Research warning banner
│   │   ├── hooks/
│   │   │   └── useWebcam.ts         # Webcam capture, ROI tracking, signal buffer
│   │   ├── api.ts                   # Backend API client
│   │   ├── App.tsx                  # Main dashboard view
│   │   └── index.css                # Premium dark glassmorphism design system
│   ├── package.json
│   └── vite.config.ts
├── models/
│   └── weights/                     # PyTorch checkpoints and scaler parameters
├── scripts/
│   ├── simulate_pipeline.py         # End-to-end benchmark & verification CLI
│   └── train_synthetic.py           # Calibration model training script
├── tests/
│   ├── test_api.py                  # API endpoint tests
│   ├── test_rppg.py                 # POS, CHROM, filters, and HR extraction tests
│   └── test_signal_quality.py       # SQI calculation & threshold tests
├── docker/
│   ├── Dockerfile.backend           # Backend container definition
│   └── Dockerfile.frontend          # Frontend production container
├── docker-compose.yml               # Multi-container orchestration
├── docs/
│   ├── architecture.md              # Technical architecture & specs
│   └── scientific_rationale.md      # Literature review & physiological basis
└── reports/
    └── technical_assessment_report.md # Experimental validation & benchmarks
```

---

## ⚡ Quick Start

### 1. Prerequisites
- Python 3.9+
- Node.js 18+ and npm
- Virtual environment tool (`venv`)

### 2. Backend Setup
```bash
cd facevital-ai

# Activate Python virtual environment
source .venv/bin/activate

# Install requirements (if not already installed)
pip install -r backend/requirements.txt

# Start FastAPI development server
uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
```
API documentation is available at `http://localhost:8000/docs`.

### 3. Frontend Setup
```bash
cd frontend

# Install dependencies
npm install

# Start Vite development server
npm run dev
```
Open `http://localhost:5173` in your browser.

---

## 🧪 Testing & Verification

### Running the Test Suite
All 34 automated unit and integration tests can be run with:
```bash
pytest tests/ -v
```

### Running the CLI Simulation Tool
Run the end-to-end simulation tool to verify rPPG extraction accuracy, SQI metrics, and neural inference latency:
```bash
python scripts/simulate_pipeline.py
```

### Training the Calibrated Biomarker Model
To train the Multi-Task 1D-CNN and save a fresh checkpoint to `models/weights/biomarker_model.pt`:
```bash
python scripts/train_synthetic.py
```

---

## 🐳 Docker Deployment

To launch the entire platform using Docker Compose:
```bash
docker-compose up --build
```
- Frontend: `http://localhost:3000`
- Backend API: `http://localhost:8000`
- API Docs: `http://localhost:8000/docs`

---

## 📚 References & Scientific Literature

1. **Wang, W., den Brinker, A. C., Stuijk, S., & de Haan, G. (2017).** Algorithmic Principles of Remote PPG. *IEEE Transactions on Biomedical Engineering*, 64(7), 1479–1491.
2. **de Haan, G., & Jeanne, V. (2013).** Robust Pulse Rate From Chrominance-Based rPPG. *IEEE Transactions on Biomedical Engineering*, 60(10), 2878–2886.
3. **Poh, M. Z., McDuff, D. J., & Picard, R. W. (2010).** Non-contact, automated cardiac pulse measurements using video imaging and blind source separation. *Optics Express*, 18(10), 10762–10774.
4. **Verkruysse, W., Svaasand, L. O., & Nelson, J. S. (2008).** Remote plethysmographic imaging using ambient light. *Optics Express*, 16(26), 21434–21445.
