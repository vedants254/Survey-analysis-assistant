# Setup Guide

## Prerequisites

### System Requirements
- Docker: 20.10+ and Docker Compose 2.0+
- Memory: Minimum 4GB RAM (8GB recommended)
- Storage: 2GB free space
- Network: Internet access for LLM API calls

### API Keys Required
At least one LLM provider API key is required:
- Google AI Studio: [Get API Key](https://makersuite.google.com/app/apikey)
- OpenAI: [Get API Key](https://platform.openai.com/api-keys)
- Anthropic: [Get API Key](https://console.anthropic.com/)
- Groq: [Get API Key](https://console.groq.com/keys)

## Quick Start

### 1. Clone Repository
```bash
git clone <repository-url>
cd LangGraph-CSV-Analysis-Task
```

### 2. Environment Setup
```bash
cp .env.example .env
nano .env
```

### 3. Configure Environment Variables
Edit .env file with your settings:
```bash
GOOGLE_API_KEY=your_google_api_key_here
DEFAULT_LLM_PROVIDER=google
DEFAULT_MODEL=gemini-2.0-flash

OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
GROQ_API_KEY=your_groq_api_key_here

POSTGRES_USER=admin
POSTGRES_PASSWORD=secure_password_2024
POSTGRES_DB=data_analysis_db
```

### 4. Start Services
```bash
docker-compose up -d
docker-compose logs -f
docker-compose ps
```

### 5. Access Application
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Documentation: http://localhost:8000/docs

### 6. Test with Sample Data
The sample_data/ directory contains test files:
- sales_q4_2024.csv
- sales_q1_2025.csv
- marketing_campaigns_2024.csv
- customer_segments_2024.csv

## Configuration

### Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| GOOGLE_API_KEY | Google Gemini API key | - | Yes* |
| OPENAI_API_KEY | OpenAI API key | - | No |
| ANTHROPIC_API_KEY | Anthropic API key | - | No |
| GROQ_API_KEY | Groq API key | - | No |
| DEFAULT_LLM_PROVIDER | Primary LLM provider | google | Yes |
| DEFAULT_MODEL | Default model name | gemini-2.0-flash | Yes |
| DATABASE_URL | PostgreSQL connection | Auto-generated | No |
| REDIS_URL | Redis connection | redis://redis:6379/0 | No |
| DEBUG | Debug mode | false | No |
| LOG_LEVEL | Logging level | INFO | No |

*At least one LLM provider API key is required

### LLM Provider Configuration

The system supports multiple LLM providers with automatic fallback:

```python
1. Google Gemini (Primary)
2. Groq (High-speed fallback)
3. OpenAI (Quality fallback)
4. Anthropic (Alternative fallback)
```

### File Upload Limits
- Max file size: 100MB per file
- Supported formats: CSV, Excel (.xlsx, .xls)
- Max files per analysis: 10 files
- Total upload limit: 500MB per session

## Development Setup

### Local Development

#### Backend Development
```bash
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

#### Frontend Development
```bash
cd frontend
npm install
npm start
```

Note: The frontend was built with AI assistance (Kiro's Claude Sonnet 4 and Gemini CLI) due to limited frontend expertise.

#### Database Migrations
```bash
alembic revision --autogenerate -m "Description"
alembic upgrade head
```

## Testing

### Running Tests

#### Backend Tests
```bash
cd backend
pytest tests/ -v --cov=app
```

#### Frontend Tests
```bash
cd frontend
npm test
```

#### Integration Tests
```bash
cd tests
python run_tests.py
```

## Deployment

### Production Deployment

#### Docker Compose (Recommended)
```bash
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

#### Environment-Specific Configurations

##### Development
```bash
DEBUG=true
LOG_LEVEL=DEBUG
```

##### Production
```bash
DEBUG=false
LOG_LEVEL=WARNING
SECURE_SSL_REDIRECT=true
```

## Troubleshooting

### Common Issues

#### 1. LLM API Errors
```bash
docker-compose logs backend | grep "API"
curl -X GET http://localhost:8000/api/v2/llm/providers
```

#### 2. Database Connection Issues
```bash
docker-compose ps db
docker-compose logs db
docker-compose exec backend python -c "from app.db import engine; print('DB OK')"
```

#### 3. File Upload Problems
```bash
docker-compose logs backend | grep "upload"
ls -la sample_data/
```

### Sample API Calls

#### Upload Files
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

#### Check Analysis Status
```bash
curl -X GET "http://localhost:8000/api/v2/analysis/{execution_id}/status"
```