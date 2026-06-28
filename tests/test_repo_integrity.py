"""Repository integrity checks against documentation and config drift."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_required_documentation_exists() -> None:
    """Core documentation referenced by README should exist."""
    required_docs = [
        "README.md",
        "ARCHITECTURE.md",
        "CONFIGURATION.md",
        "QUICKSTART.md",
        "SAFETY.md",
        "CONTRIBUTING.md",
        "SECURITY.md",
        "CHANGELOG.md",
        "LICENSE",
    ]
    for doc_path in required_docs:
        assert (REPO_ROOT / doc_path).exists(), f"missing required doc: {doc_path}"


def test_readme_contains_positioning_and_operational_sections() -> None:
    """README should retain project positioning and operator guidance."""
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    required_sections = [
        "Problems It Solves",
        "Demo Gallery",
        "Quality Gates",
        "Routing Strategies",
        "PYTHONPATH=src uvicorn api.main:app",
        "Apache-2.0",
    ]
    for section in required_sections:
        assert section in readme, f"README missing expected section or marker: {section}"


def test_ci_targets_supported_python_versions() -> None:
    """CI should not test unsupported Python versions."""
    ci_config = (REPO_ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    assert "3.10" not in ci_config
    assert "3.11" in ci_config
    assert "ruff check src/" in ci_config
    assert "mypy src/" in ci_config


def test_nexus_env_example_keys_are_documented() -> None:
    """Every NEXUS-prefixed environment example key should be documented."""
    env_example = (REPO_ROOT / ".env.example").read_text(encoding="utf-8")
    configuration_doc = (REPO_ROOT / "CONFIGURATION.md").read_text(encoding="utf-8")
    nexus_keys = {
        line.split("=", maxsplit=1)[0]
        for line in env_example.splitlines()
        if line.startswith("NEXUS_")
    }

    missing_keys = sorted(key for key in nexus_keys if key not in configuration_doc)
    assert missing_keys == []


def test_dockerfile_uses_production_dependencies() -> None:
    """Production image should not install dev extras."""
    dockerfile = (REPO_ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert ".[dev]" not in dockerfile
    assert "pip install" in dockerfile


def test_readme_test_badge_matches_collected_tests() -> None:
    """README test badge count must match collected pytest functions."""
    import ast

    test_count = sum(
        1
        for path in (REPO_ROOT / "tests").glob("test_*.py")
        for node in ast.parse(path.read_text(encoding="utf-8")).body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name.startswith("test_")
    )
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert f"tests-{test_count}%20passing" in readme
