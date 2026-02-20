# Iikshana: ADA-Compliant Courtroom Visual Aid System

Gemini-powered agentic AI system for multilingual courtroom accessibility designed for blind individuals.

## Team - Group 16, IE7374 MLOps

- Aditya Vasisht
- Akshata Kumble  
- Amit Karanth Gurpur
- Rohit Abhijit Kulkarni
- Shridhar Sunilkumar Pol
- Suraj Patel Muthe Gowda

## Quick Start

Documentation coming soon.
```

### **.gitignore**
```
# Python
__pycache__/
*.py[cod]
venv/
env/
*.egg-info/

# Node
node_modules/
build/
dist/

# Environment
.env
.env.local

# Data
data/raw/*
data/processed/*
!data/*/.gitkeep

# Logs
*.log
logs/

# API Keys
credentials.json

## Repository Structure
```
iikshana-courtroom-accessibility/
├── backend/
│   └── src/
│       ├── agents/          # 6 AI agents + orchestrator
│       ├── services/        # Gemini, TTS, WebSocket
│       ├── api/             # Routes, handlers
│       └── main.py
├── frontend/
│   └── src/
│       ├── components/      # React UI (WCAG AAA)
│       ├── services/        # API clients
│       ├── hooks/           # Custom hooks
│       └── App.tsx
<<<<<<< HEAD
├── data/
│   ├── legal_glossary/      # 500+ legal terms
│   └── raw/                 # Evaluation datasets
├── data-pipeline/           # Data acquisition, preprocessing, validation, evaluation (Airflow + DVC)
=======
├── airflow/                 # Airflow DAGs + Docker (orchestration); mounts data-pipeline
├── data-pipeline/           # Data acquisition, preprocessing, validation, evaluation (scripts + config)
├── data/
│   ├── legal_glossary/      # 500+ legal terms
│   └── raw/                 # Evaluation datasets
>>>>>>> origin/data_preprocessing
├── docs/                    # Architecture, API docs
└── config/                  # Environment configs
```