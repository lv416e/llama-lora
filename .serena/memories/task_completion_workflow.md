# Task Completion Workflow

## Before Committing Changes

### 1. Code Quality Checks
```bash
# Lint the code - fix any issues
uv run ruff check .

# Format the code
uv run ruff format
```

### 2. Run Tests
```bash
# Run all tests
pytest tests/

# Run specific tests if relevant
pytest tests/test_config_validation.py -v
```

### 3. Validate Configuration (if config changes)
```bash
# Validate default configuration
uv run python -m llama_lora.validate

# Validate specific experiment configs
uv run python -m llama_lora.validate +experiment=quick_test
```

### 4. Smoke Test (for ML code changes)
```bash
# Quick end-to-end test (CPU-friendly)
uv run python -m llama_lora.train +experiment=quick_test
```

## Git Workflow

### Commit Messages
Follow Conventional Commits format:
- `feat:` - new feature
- `fix:` - bug fix
- `docs:` - documentation changes
- `chore:` - maintenance tasks
- `refactor:` - code refactoring

Example: `feat: add DoRA support to training pipeline`

### Pull Request Requirements
- Clear summary and rationale
- Link to related issues (`Fixes #123`)
- Describe what changed and how to test
- Note resource requirements (CPU/GPU)
- Update README.md if commands/UX changed
- Include screenshots/logs if helpful

## Security Considerations
- Never commit secrets or tokens
- Ensure `huggingface-cli login` is documented for gated models
- Avoid committing large model artifacts in `outputs/`
- Check that sensitive paths are in `.gitignore`

## Documentation Updates
- Update README.md for user-facing changes
- Update AGENTS.md for development workflow changes
- Keep docstrings current for complex functions