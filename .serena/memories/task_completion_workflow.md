# Task Completion Workflow

## 🎯 Code Quality Checklist (MANDATORY)

### 1. Code Formatting & Linting
```bash
# ALWAYS run before any commit or task completion
uv run ruff check --fix .     # Auto-fix linting issues
uv run ruff format .          # Format code consistently

# Verify clean state
uv run ruff check .           # Should show "All checks passed!"
```

### 2. Type Safety & Validation
```bash
# Validate configuration schemas
uv run python -m llama_lora.validate

# Test configuration parsing
uv run python -c "
from config.schema import HydraConfig
config = HydraConfig()
print('✅ Configuration validation passed')
"
```

## 🧪 Testing Requirements

### After Code Changes
```bash
# Run relevant tests based on changes
pytest tests/test_tokenizer_utils.py      # If utils/common.py changed
pytest tests/test_config_validation.py    # If config/schema.py changed
pytest tests/test_path_manager.py         # If file operations changed

# Full test suite for major changes
pytest tests/ -v
```

### After Configuration Changes
```bash
# Test all experiment configurations
uv run python -m llama_lora.validate +experiment=quick_test
uv run python -m llama_lora.validate +experiment=full_training

# Quick smoke test
uv run python -m llama_lora.train +experiment=quick_test training.epochs=1
```

## 🚀 Functional Verification

### After Training Pipeline Changes
```bash
# End-to-end smoke test (REQUIRED)
echo "=== 1. Baseline Check ==="
uv run python -m llama_lora.baseline

echo "=== 2. Training Test ==="
uv run python -m llama_lora.train +experiment=quick_test

echo "=== 3. Inference Test ==="
uv run python -m llama_lora.infer "Test prompt for validation"

echo "=== 4. Merge Test ==="
uv run python -m llama_lora.merge

echo "=== 5. Output Verification ==="
ls -la outputs/adapter/     # Should contain adapter files
ls -la outputs/merged/      # Should contain merged model
```

### After Inference Changes
```bash
# Test inference with different parameters
uv run python -m llama_lora.infer "Simple test" --max_new_tokens 32
uv run python -m llama_lora.infer "Complex reasoning task" --temperature 0.1
```

### After Configuration System Changes
```bash
# Test override mechanisms
uv run python -m llama_lora.train training.lr=1e-5 --dry-run
uv run python -m llama_lora.train model.use_dora=true --dry-run
```

## 📁 File System Verification

### Output Directory Structure
```bash
# Verify expected output structure after training
tree outputs/ -I "__pycache__"

# Expected structure:
# outputs/
# ├── adapter/
# │   ├── adapter_config.json
# │   └── adapter_model.safetensors
# ├── tokenizer/
# │   ├── tokenizer_config.json
# │   └── tokenizer.json
# ├── runs/                    # TensorBoard logs
# └── merged/                  # After merge operation
```

### Cleanup Verification
```bash
# Ensure no temporary files left behind
find . -name "*.tmp" -o -name "*.log" -o -name "core.*" | head -10
find . -type d -name "__pycache__" | head -5
```

## 🔍 System Integration Checks

### Device Compatibility
```bash
# Verify device detection works correctly
uv run python -c "
from src.llama_lora.utils.common import DeviceManager
device = DeviceManager.detect_device()
print(f'✅ Detected device: {device}')

fp16, bf16 = DeviceManager.setup_device_specific_settings(device)
print(f'✅ Device settings: fp16={fp16}, bf16={bf16}')
"
```

### Memory Management
```bash
# Check for memory leaks during training
# (Only run if memory issues suspected)
uv run python -c "
import torch
import gc
print(f'Initial memory: {torch.cuda.memory_allocated() if torch.cuda.is_available() else 0}')
# Run small training step here
gc.collect()
torch.cuda.empty_cache() if torch.cuda.is_available() else None
print(f'Final memory: {torch.cuda.memory_allocated() if torch.cuda.is_available() else 0}')
"
```

## 📝 Documentation Updates

### When Code Structure Changes
- Update README.md if new commands or workflows added
- Update configuration examples if new parameters added
- Update memory files if architecture significantly changed

### When Adding New Features
```bash
# Ensure new modules have proper docstrings
grep -r "\"\"\"" src/llama_lora/ --include="*.py" | wc -l

# Check for missing type hints
uv run python -c "
import ast
import sys
from pathlib import Path

def check_type_hints(file_path):
    with open(file_path) as f:
        tree = ast.parse(f.read())
    
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            if not node.returns and node.name != '__init__':
                print(f'Missing return type: {file_path}:{node.lineno} {node.name}')

for py_file in Path('src/llama_lora').rglob('*.py'):
    check_type_hints(py_file)
"
```

## ⚠️ Pre-Commit Checklist

### MANDATORY Steps (Never Skip)
1. ✅ Run `uv run ruff check --fix .`
2. ✅ Run `uv run ruff format .`
3. ✅ Verify `uv run ruff check .` passes
4. ✅ Run basic functional test (smoke test)
5. ✅ Check no unintended files in git staging

### For Significant Changes
1. ✅ Run full test suite: `pytest tests/`
2. ✅ Run end-to-end workflow test
3. ✅ Validate all experiment configurations
4. ✅ Update relevant documentation

## 🚨 Failure Response Procedures

### If Tests Fail
```bash
# Don't commit until all tests pass
pytest tests/ -v --tb=short     # Get detailed failure info
pytest tests/failing_test.py -s # Debug specific test

# Fix issues and re-run
pytest tests/test_that_failed.py -v
```

### If Linting Fails
```bash
# Check what ruff wants to change
uv run ruff check . --diff

# Apply fixes
uv run ruff check --fix .
uv run ruff format .

# Manually review and fix remaining issues
uv run ruff check .
```

### If Functional Tests Fail
```bash
# Check logs for error details
tail -50 /tmp/llama_lora_debug.log

# Test components individually
uv run python -m llama_lora.validate
uv run python -c "from src.llama_lora.utils.common import setup_logging; setup_logging()"
```

## 📊 Success Criteria

### All Tasks Must Meet
- ✅ Ruff checks pass completely
- ✅ No Python import errors
- ✅ Configuration validation passes
- ✅ Basic smoke test completes successfully

### Significant Changes Must Also Meet
- ✅ Full test suite passes
- ✅ End-to-end workflow completes
- ✅ Memory usage remains stable
- ✅ Device compatibility maintained