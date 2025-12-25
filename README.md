# Analysis assistant

## Overview

A production-grade Historical Multi-Table Data Analysis Platform built with LangGraph that provides conversational AI-powered analysis of CSV/Excel files with temporal comparison capabilities. The system uses an 8-node workflow orchestration engine to process multiple data files, align time series, generate Python analysis code, and provide intelligent insights. Setup locally !
 
## Features

### Core Capabilities
- Multi-File Analysis: Process multiple CSV/Excel files simultaneously
- Temporal Comparisons: Month-over-Month (MoM), Quarter-over-Quarter (QoQ), Year-over-Year (YoY)
- Cross-Table Operations: Join and analyze data across different files
- AI-Powered Code Generation: Dynamic Python code generation using LLMs
- Real-time Progress Tracking: WebSocket-based progress updates
- Error Recovery: Intelligent retry mechanisms with context-aware error handling

### Advanced Features
- JSON Metadata Parsing: Extract and analyze nested JSON fields
- Trend Analysis: Automated pattern detection and anomaly identification
- Multi-LLM Support: Google Gemini, OpenAI, Anthropic, Groq with fallback
- Safe Code Execution: Sandboxed Python execution with security validation
- Graceful Degradation: Fault-tolerant operation with fallback mechanisms

## Technology Stack

### Backend
- FastAPI: Modern Python web framework
- LangGraph: Workflow orchestration and state management
- SQLAlchemy: ORM with async support
- Celery: Distributed task queue
- Pandas: Data manipulation and analysis
- Psycopg: PostgreSQL async driver

### Frontend
- React 18: Modern UI framework
- Material-UI: Component library
- Axios: HTTP client
- Recharts: Data visualization
- React Router: Navigation

Note: The frontend implementation was developed with assistance from AI tools (Kiro's Claude Sonnet 4 and Gemini CLI) as the author is not extensively familiar with modern frontend frameworks and React ecosystem.

### Infrastructure
- Docker: Containerization
- PostgreSQL: Primary database
- Redis: Caching and message broker
- Nginx: Reverse proxy (production)

### AI/ML
- Google Gemini: Primary LLM provider
- OpenAI GPT: Alternative LLM provider
- Anthropic Claude: Alternative LLM provider
- Groq: High-speed inference provider

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed system architecture, workflow diagrams, and component specifications.

## Quick Start

See [SETUP.md](SETUP.md) for detailed setup instructions, configuration, and troubleshooting.

Quick Commands:
```bash
git clone <repository-url>
cd LangGraph-CSV-Analysis-Task
cp .env.example .env

docker-compose up -d
```

Access:
- Frontend: http://localhost:3000
- Backend: http://localhost:8000
- API Docs: http://localhost:8000/docs

## API Documentation

### Core Endpoints

#### File Upload
```bash
curl -X POST "http://localhost:8000/api/files/upload" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@sample_data/sales_q4_2024.csv"
```

#### Start Analysis
```bash
curl -X POST "http://localhost:8000/api/v2/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Compare Q4 2024 vs Q1 2025 sales performance",
    "file_ids": ["file1_id", "file2_id"],
    "session_id": "test_session_123"
  }'
```

#### Check Status
```bash
curl -X GET "http://localhost:8000/api/v2/analysis/{execution_id}/status"
```

#### WebSocket Progress
```javascript
ws://localhost:8000/ws/analysis/{execution_id}
```

## Project Structure

```
LangGraph-CSV-Analysis-Task/
├── frontend/
├── backend/
├── langgraph/
├── workers/
├── db/
├── tests/
├── sample_data/
├── docker-compose.yml
├── SETUP.md
├── ARCHITECTURE.md
└── README.md
```

## Testing

```bash
cd backend && pytest tests/ -v
cd frontend && npm test
cd tests && python run_tests.py
```

## Deployment

### Development
```bash
docker-compose up -d
```

### Production
```bash
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

## Troubleshooting

Common issues and solutions are documented in [SETUP.md](SETUP.md).

Quick Checks:
```bash
docker-compose ps
docker-compose logs backend
curl http://localhost:8000/health
```

## Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/name`
3. Commit changes: `git commit -m 'Add feature'`
4. Push to branch: `git push origin feature/name`
5. Open Pull Request
