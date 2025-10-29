# Testing Suite

Comprehensive testing suite for the LangGraph CSV Analysis platform.

## Overview

This testing suite provides comprehensive coverage for:
- API endpoints and integration
- LangGraph workflow functionality  
- Error handling and edge cases
- Installation and system requirements
- Performance and data processing

## Test Structure

```
tests/
├── conftest.py           # Test configuration and fixtures
├── test_api.py           # API endpoint tests
├── test_integration.py   # Workflow integration tests  
├── test_installation.py  # System and installation tests
├── run_tests.py          # Test runner script
├── requirements.txt      # Test dependencies
├── pytest.ini           # Pytest configuration
└── README.md            # This file
```

## Quick Start

### Install Test Dependencies

```bash
# From the tests directory
pip install -r requirements.txt
```

### Run All Tests

```bash
# Using the test runner (recommended)
python run_tests.py

# Or using pytest directly
python -m pytest
```

## Test Categories

### API Tests (`test_api.py`)
Tests all REST API endpoints and WebSocket functionality:
- Health check endpoints
- File upload and management
- Analysis endpoints (simple and comprehensive)
- Session management
- Error handling
- Authentication and security

### Integration Tests (`test_integration.py`)
Tests the complete LangGraph workflow:
- Individual workflow nodes
- End-to-end workflow execution
- Multi-file analysis scenarios
- Temporal data alignment
- Error handling in workflows
- Data processing capabilities

### Installation Tests (`test_installation.py`)
Tests system requirements and installation process:
- Python version and dependencies
- Node.js and npm availability
- Environment file setup
- Deployment script functionality
- Service health checks
- Sample data validation

## Running Specific Tests

### By Test Type
```bash
# API tests only
python run_tests.py --type api

# Integration tests only  
python run_tests.py --type integration

# System/installation tests only
python run_tests.py --type system

# Quick tests (exclude slow tests)
python run_tests.py --type quick
```

### By Test File
```bash
# Run specific test file
python -m pytest test_api.py

# Run specific test class
python -m pytest test_api.py::TestHealthAPI

# Run specific test method
python -m pytest test_api.py::TestHealthAPI::test_health_check
```

### With Coverage
```bash
# Terminal coverage report
python run_tests.py --coverage

# HTML coverage report
python run_tests.py --coverage --html

# XML coverage report (for CI/CD)
python run_tests.py --coverage --xml
```

## Test Configuration

### Environment Variables
Tests use these environment variables:
- `TESTING=true` - Enables test mode
- `OPENAI_API_KEY=test-key-123` - Mock API key for testing
- `DATABASE_URL=sqlite:///test_db.sqlite` - Test database
- `LOG_LEVEL=WARNING` - Reduced logging for tests

### Fixtures and Mocks
Key fixtures provided in `conftest.py`:
- `test_client` - FastAPI test client
- `sample_csv_data` - Sample pandas DataFrame
- `sample_csv_file` - Temporary CSV file
- `mock_openai_client` - Mocked LLM client
- `mock_database` - Mocked database session
- `mock_workflow` - Mocked LangGraph workflow

## Continuous Integration

For CI/CD pipelines:

```bash
# Install test dependencies
pip install -r tests/requirements.txt

# Run tests with XML coverage for CI
cd tests && python run_tests.py --coverage --xml

# Run quick tests only (for faster CI)
cd tests && python run_tests.py --type quick
```

## Writing New Tests

### Test File Naming
- `test_*.py` - Test files
- `Test*` - Test classes
- `test_*` - Test methods

### Example Test
```python
import pytest
from conftest import TestDataGenerator

class TestMyFeature:
    """Test my new feature."""
    
    def test_basic_functionality(self, test_client):
        """Test basic functionality."""
        response = test_client.get("/my-endpoint")
        assert response.status_code == 200
        
    @pytest.mark.asyncio
    async def test_async_functionality(self, mock_workflow):
        """Test async functionality."""
        result = await my_async_function()
        assert result is not None
        
    @pytest.mark.slow
    def test_slow_operation(self):
        """Test that takes a long time."""
        # This test will be skipped in quick test runs
        pass
```

### Test Markers
Available pytest markers:
- `@pytest.mark.slow` - Slow tests
- `@pytest.mark.integration` - Integration tests  
- `@pytest.mark.api` - API tests
- `@pytest.mark.unit` - Unit tests
- `@pytest.mark.system` - System tests

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Ensure test dependencies are installed
   pip install -r tests/requirements.txt
   ```

2. **Path Issues**
   ```bash
   # Run tests from the tests directory
   cd tests && python -m pytest
   ```

3. **Database Conflicts**
   ```bash
   # Remove test database
   rm test_db.sqlite
   ```

4. **Port Conflicts**
   - Tests use port 8001 for backend testing
   - Ensure port 8001 is available

### Debug Mode
```bash
# Run with verbose output and no capture
python -m pytest -v -s test_api.py::TestHealthAPI::test_health_check
```

### Performance Testing
```bash
# Run with timing information
python -m pytest --durations=10
```

## Coverage Reports

After running tests with coverage:
- Terminal report shows missing lines
- HTML report available in `htmlcov/index.html`
- XML report available in `coverage.xml`

Target coverage goals:
- Overall: 80%+
- Critical paths: 95%+
- API endpoints: 90%+

## Contributing

When adding new features:
1. Write tests first (TDD approach)
2. Ensure existing tests pass
3. Add appropriate test markers
4. Update this README if needed
5. Maintain coverage above 80%