# tests/test_installation.py

"""
Installation and system tests for the LangGraph CSV Analysis platform.
Tests the basic installation flow and system requirements for the new architecture.
"""

import subprocess
import sys
import pytest
from pathlib import Path

class TestSystemRequirements:
    """Test system requirements and dependencies."""

    def test_python_version(self):
        """Test that Python version is 3.8 or higher."""
        major, minor = sys.version_info[:2]
        assert major >= 3, f"Python 3.x required, got {major}.{minor}"
        assert minor >= 8, f"Python 3.8+ required, got 3.{minor}"

    def test_docker_and_docker_compose_availability(self):
        """Test that Docker and Docker Compose are available."""
        try:
            docker_result = subprocess.run(["docker", "--version"], capture_output=True, text=True, check=True)
            assert "Docker version" in docker_result.stdout
        except (subprocess.CalledProcessError, FileNotFoundError):
            pytest.fail("Docker is not installed or not in PATH. It is required to run the application.")

        try:
            compose_result = subprocess.run(["docker-compose", "--version"], capture_output=True, text=True, check=True)
            assert "Docker Compose version" in compose_result.stdout
        except (subprocess.CalledProcessError, FileNotFoundError):
            pytest.fail("Docker Compose is not installed or not in PATH. It is required to run the application.")

class TestProjectStructure:
    """Test the basic project structure and configuration files."""

    @pytest.fixture
    def project_root(self):
        """Get the project root directory."""
        return Path(__file__).parent.parent

    def test_docker_compose_exists(self, project_root):
        """Test that docker-compose.yml file exists."""
        assert (project_root / "docker-compose.yml").exists(), "docker-compose.yml is missing."

    def test_backend_requirements_format(self, project_root):
        """Test that backend/requirements.txt is properly formatted."""
        requirements_file = project_root / "backend" / "requirements.txt"
        assert requirements_file.exists(), "backend/requirements.txt not found"

        content = requirements_file.read_text()
        assert "fastapi" in content
        assert "celery" in content
        assert "sqlalchemy" in content
        assert "psycopg2-binary" in content

    def test_streamlit_requirements_format(self, project_root):
        """Test that streamlit_app/requirements.txt is properly formatted."""
        requirements_file = project_root / "streamlit_app" / "requirements.txt"
        assert requirements_file.exists(), "streamlit_app/requirements.txt not found"

        content = requirements_file.read_text()
        assert "streamlit" in content
        assert "requests" in content

class TestDocumentation:
    """Test essential documentation files."""

    def test_readme_exists(self, project_root):
        """Test that README file exists and is not empty."""
        readme_file = project_root / "README.md"
        assert readme_file.exists(), "README.md not found"
        content = readme_file.read_text()
        assert len(content) > 100, "README.md appears to be empty."
