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

## Final Repository Structure
```
iikshana-courtroom-accessibility/
├── .github/
│   ├── workflows/
│   │   ├── ci.yml
│   │   ├── deploy-backend.yml
│   │   └── deploy-frontend.yml
│   └── ISSUE_TEMPLATE/
│       ├── bug_report.md
│       └── feature_request.md
├── backend/
│   ├── src/
│   │   ├── agents/
│   │   │   ├── __init__.py
│   │   │   ├── orchestrator.py
│   │   │   ├── audio_intelligence.py
│   │   │   ├── translation_agent.py
│   │   │   ├── legal_glossary_guardian.py
│   │   │   ├── vision_agent.py
│   │   │   ├── speech_synthesis.py
│   │   │   └── context_manager.py
│   │   ├── services/
│   │   │   ├── __init__.py
│   │   │   ├── gemini_service.py
│   │   │   ├── tts_service.py
│   │   │   └── websocket_service.py
│   │   ├── api/
│   │   │   ├── __init__.py
│   │   │   ├── routes.py
│   │   │   └── websocket_handler.py
│   │   ├── models/
│   │   │   ├── __init__.py
│   │   │   ├── schemas.py
│   │   │   └── enums.py
│   │   ├── utils/
│   │   │   ├── __init__.py
│   │   │   ├── audio_processing.py
│   │   │   ├── logger.py
│   │   │   └── config.py
│   │   └── main.py
│   ├── tests/
│   ├── requirements.txt
│   └── README.md
├── frontend/
│   ├── public/
│   │   ├── index.html
│   │   └── manifest.json
│   ├── src/
│   │   ├── components/
│   │   │   ├── AudioCapture.tsx
│   │   │   ├── TranscriptDisplay.tsx
│   │   │   ├── ControlPanel.tsx
│   │   │   ├── ImageViewer.tsx
│   │   │   └── AccessibilityControls.tsx
│   │   ├── services/
│   │   │   ├── websocket.ts
│   │   │   ├── api.ts
│   │   │   └── audio.ts
│   │   ├── hooks/
│   │   │   ├── useWebSocket.ts
│   │   │   ├── useAudioCapture.ts
│   │   │   └── useKeyboardShortcuts.ts
│   │   ├── utils/
│   │   │   ├── constants.ts
│   │   │   └── helpers.ts
│   │   ├── types/
│   │   │   └── index.ts
│   │   ├── App.tsx
│   │   └── index.tsx
│   ├── package.json
│   ├── tsconfig.json
│   └── README.md
├── data/
│   ├── raw/.gitkeep
│   ├── processed/.gitkeep
│   ├── legal_glossary/
│   │   └── legal_terms.json
│   └── README.md
├── docs/
│   ├── architecture.md
│   ├── api_documentation.md
│   ├── deployment_guide.md
│   ├── user_manual.md
│   └── agent_specifications.md
├── config/
│   ├── development.yaml
│   ├── production.yaml
│   └── testing.yaml
├── scripts/
│   ├── setup.sh
│   ├── deploy.sh
│   └── test.sh
├── logs/.gitkeep
├── .gitignore
├── .env.example
├── README.md
├── CONTRIBUTING.md
└── LICENSE