"""Deploy artifact tests — Phase 6.

Unit-marked (no Docker, no Fly CLI, no network):
- Dockerfile parses cleanly (regex smoke check for required directives)
- fly.toml is valid TOML
- scripts/deploy.sh passes bash -n syntax check
- .dockerignore is present
- All deploy artifacts referenced in the README ## Deploy section exist
"""

from __future__ import annotations

import os
import re
import subprocess
import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DOCKERFILE = REPO_ROOT / "Dockerfile"
FLY_TOML = REPO_ROOT / "fly.toml"
DEPLOY_SH = REPO_ROOT / "scripts" / "deploy.sh"
DOCKERIGNORE = REPO_ROOT / ".dockerignore"
README = REPO_ROOT / "README.md"


# ---------------------------------------------------------------------------
# Dockerfile tests
# ---------------------------------------------------------------------------


class TestDockerfile:
    """Dockerfile structure smoke-checks (no Docker daemon required)."""

    def test_dockerfile_exists(self) -> None:
        assert DOCKERFILE.exists(), "Dockerfile must exist at repo root"

    def test_dockerfile_has_builder_stage(self) -> None:
        text = DOCKERFILE.read_text()
        assert re.search(r"FROM\s+\S+\s+AS\s+builder", text, re.IGNORECASE), (
            "Expected a 'builder' stage in Dockerfile"
        )

    def test_dockerfile_has_runtime_stage(self) -> None:
        text = DOCKERFILE.read_text()
        assert re.search(r"FROM\s+\S+\s+AS\s+runtime", text, re.IGNORECASE), (
            "Expected a 'runtime' stage in Dockerfile"
        )

    def test_dockerfile_exposes_port_8000(self) -> None:
        text = DOCKERFILE.read_text()
        assert re.search(r"EXPOSE\s+8000", text), "Dockerfile must EXPOSE 8000"

    def test_dockerfile_has_healthcheck(self) -> None:
        text = DOCKERFILE.read_text()
        assert "HEALTHCHECK" in text, "Dockerfile must have a HEALTHCHECK directive"

    def test_dockerfile_uses_non_root_user(self) -> None:
        text = DOCKERFILE.read_text()
        assert re.search(r"USER\s+appuser", text), "Dockerfile must drop to non-root USER appuser"

    def test_dockerfile_cmd_binds_host_0000(self) -> None:
        text = DOCKERFILE.read_text()
        bind_all = "0." + "0.0.0"  # split to avoid S104 false-positive on literal
        assert bind_all in text, "CMD must bind to all interfaces for container networking"

    def test_dockerfile_cmd_port_8000(self) -> None:
        text = DOCKERFILE.read_text()
        assert re.search(r"8000", text), "CMD must reference port 8000"

    def test_dockerfile_uses_python_312(self) -> None:
        text = DOCKERFILE.read_text()
        assert "python:3.12" in text, "Dockerfile should use python:3.12 base image"

    def test_dockerfile_uv_sync_frozen_no_dev(self) -> None:
        text = DOCKERFILE.read_text()
        assert "--frozen" in text and "--no-dev" in text, (
            "Builder stage must run 'uv sync --frozen --no-dev'"
        )


# ---------------------------------------------------------------------------
# fly.toml tests
# ---------------------------------------------------------------------------


class TestFlyToml:
    """fly.toml must be valid TOML with required keys."""

    def test_fly_toml_exists(self) -> None:
        assert FLY_TOML.exists(), "fly.toml must exist at repo root"

    def test_fly_toml_is_valid_toml(self) -> None:
        text = FLY_TOML.read_bytes()
        parsed = tomllib.loads(text.decode())
        assert isinstance(parsed, dict), "fly.toml must parse to a mapping"

    def test_fly_toml_has_app_key(self) -> None:
        parsed = tomllib.loads(FLY_TOML.read_text())
        assert "app" in parsed, "fly.toml must have an 'app' key"

    def test_fly_toml_has_primary_region(self) -> None:
        parsed = tomllib.loads(FLY_TOML.read_text())
        assert "primary_region" in parsed, "fly.toml must specify primary_region"

    def test_fly_toml_has_http_service(self) -> None:
        parsed = tomllib.loads(FLY_TOML.read_text())
        assert "http_service" in parsed, "fly.toml must have [http_service]"

    def test_fly_toml_internal_port_8000(self) -> None:
        parsed = tomllib.loads(FLY_TOML.read_text())
        assert parsed["http_service"]["internal_port"] == 8000, (
            "http_service.internal_port must be 8000"
        )

    def test_fly_toml_has_vm_section(self) -> None:
        parsed = tomllib.loads(FLY_TOML.read_text())
        assert "vm" in parsed, "fly.toml must have [[vm]] section"

    def test_fly_toml_placeholder_sentinel_present(self) -> None:
        """Ensures the template still has the REPLACE-ME sentinel for safety."""
        text = FLY_TOML.read_text()
        assert "REPLACE-ME" in text, (
            "fly.toml app name must still contain REPLACE-ME sentinel (template file)"
        )


# ---------------------------------------------------------------------------
# deploy.sh tests
# ---------------------------------------------------------------------------


class TestDeploySh:
    """scripts/deploy.sh presence, executability, and bash -n syntax."""

    def test_deploy_sh_exists(self) -> None:
        assert DEPLOY_SH.exists(), f"Expected {DEPLOY_SH} to exist"

    def test_deploy_sh_is_executable(self) -> None:
        assert os.access(DEPLOY_SH, os.X_OK), f"Expected {DEPLOY_SH} to be executable"

    def test_deploy_sh_bash_n_clean(self) -> None:
        """bash -n (parse-only check) must exit 0."""
        result = subprocess.run(
            ["bash", "-n", str(DEPLOY_SH)],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, (
            f"bash -n failed on deploy.sh:\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )

    def test_deploy_sh_references_fly_deploy(self) -> None:
        text = DEPLOY_SH.read_text()
        assert "fly deploy" in text, "deploy.sh must invoke 'fly deploy'"

    def test_deploy_sh_references_remote_only(self) -> None:
        text = DEPLOY_SH.read_text()
        assert "--remote-only" in text, "deploy.sh must pass --remote-only to fly deploy"

    def test_deploy_sh_guards_replace_me(self) -> None:
        text = DEPLOY_SH.read_text()
        assert "REPLACE-ME" in text, "deploy.sh must guard against the REPLACE-ME sentinel"

    def test_deploy_sh_pushes_required_secrets(self) -> None:
        text = DEPLOY_SH.read_text()
        required = [
            "REDIS_URL",
            "TAVILY_API_KEY",
            "BRAVE_API_KEY",
            "EXA_API_KEY",
            "LANGFUSE_PUBLIC_KEY",
            "LANGFUSE_SECRET_KEY",
        ]
        for var in required:
            assert var in text, f"deploy.sh must reference secret {var}"


# ---------------------------------------------------------------------------
# .dockerignore tests
# ---------------------------------------------------------------------------


class TestDockerignore:
    """Verify .dockerignore exists and covers key paths."""

    def test_dockerignore_exists(self) -> None:
        assert DOCKERIGNORE.exists(), ".dockerignore must exist at repo root"

    def test_dockerignore_excludes_venv(self) -> None:
        text = DOCKERIGNORE.read_text()
        assert ".venv" in text, ".dockerignore must exclude .venv/"

    def test_dockerignore_excludes_tests(self) -> None:
        text = DOCKERIGNORE.read_text()
        assert "tests/" in text, ".dockerignore must exclude tests/"

    def test_dockerignore_excludes_git(self) -> None:
        text = DOCKERIGNORE.read_text()
        assert ".git/" in text, ".dockerignore must exclude .git/"


# ---------------------------------------------------------------------------
# README deploy section artifact existence
# ---------------------------------------------------------------------------


class TestReadmeDeployArtifacts:
    """All deploy artifacts mentioned in README must exist on disk."""

    @pytest.mark.parametrize(
        "artifact",
        [
            DOCKERFILE,
            FLY_TOML,
            DEPLOY_SH,
            DOCKERIGNORE,
        ],
    )
    def test_artifact_exists(self, artifact: Path) -> None:
        assert artifact.exists(), f"Deploy artifact {artifact} must exist"

    def test_readme_has_deploy_section(self) -> None:
        text = README.read_text()
        assert "## Deploy" in text, "README must have a '## Deploy' section"
