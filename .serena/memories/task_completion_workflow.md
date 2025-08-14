# Task Completion Workflow

## When a Task is Completed

### 1. Code Quality Checks (REQUIRED)
Always run these commands before considering a task complete:

```bash
# Format code with Ruff
ruff format .

# Check for style issues
ruff check .

# Fix auto-fixable issues
ruff check --fix .
```

### 2. Testing (REQUIRED)
Verify that all tests pass:

```bash
# Run all tests
pytest tests/

# Run with verbose output if debugging needed
pytest tests/ -v

# For specific modules touched
pytest tests/test_<relevant_module>.py -v
```

### 3. Configuration Validation (For Config Changes)
If configuration was modified:

```bash
# Validate default config
uv run python -m llama_lora.validate

# Validate experiment configs
uv run python -m llama_lora.validate +experiment=quick_test
uv run python -m llama_lora.validate +experiment=default-dora
```

### 4. Smoke Test (For Major Changes)
Run a quick end-to-end test:

```bash
# Quick training test
uv run python -m llama_lora.train +experiment=quick_test

# Verify inference works
uv run python -m llama_lora.infer "Test prompt"
```

### 5. Documentation Updates
- Update docstrings if function signatures changed
- Update README.md if user-facing features changed
- Ensure type hints are complete and accurate

### 6. Clean Up
```bash
# Remove Python cache files
find . -type d -name __pycache__ -exec rm -rf {} +

# Clean up any test outputs
rm -rf outputs/experiments/test_*
```

### 7. Final Verification Checklist
Before marking complete, verify:
- [ ] Code is formatted (ruff format)
- [ ] No linting errors (ruff check)
- [ ] All tests pass (pytest)
- [ ] Type hints are present
- [ ] Docstrings are updated
- [ ] No debug print statements left
- [ ] No TODO comments without issues created
- [ ] Configuration validates
- [ ] Changes work end-to-end

### 8. Git Workflow (When Requested)
Only if explicitly asked to commit:

```bash
# Check what changed
git status
git diff

# Stage and commit
git add -A
git commit -m "type: description"
```

## Error Recovery Workflow

If any step fails:

1. **Linting Errors**: Fix manually or use `ruff check --fix`
2. **Test Failures**: Debug and fix, re-run specific test
3. **Config Validation**: Check schema.py for constraints
4. **Runtime Errors**: Check error messages, fix, and re-test

## Performance Considerations

After implementation:
- Check memory usage isn't increased significantly
- Verify training still runs on target hardware
- Ensure no performance regressions in critical paths

## Important Notes

- **NEVER** skip the ruff format/check steps
- **ALWAYS** run tests after making changes
- **VERIFY** configuration changes don't break existing presets
- **TEST** on small data first before full runs
- **DOCUMENT** any new parameters or features