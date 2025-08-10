# Darwin (macOS) System Information

## System Environment
- **Platform**: Darwin (macOS)
- **Python Version**: 3.12+ (enforced via .python-version)
- **Package Manager**: uv (preferred over pip for performance)

## macOS-Specific Optimizations

### Apple Silicon (M1/M2/M3) Support
- **MPS Backend**: Metal Performance Shaders for GPU acceleration
- **Memory Management**: Unified memory architecture optimization
- **Mixed Precision**: Disabled on MPS (fp32 only for stability)

### Device Detection Priority
```python
# Automatic fallback chain for macOS
if torch.cuda.is_available():           # External GPU (rare)
    device = "cuda"
elif torch.backends.mps.is_available(): # Apple Silicon
    device = "mps"  
else:
    device = "cpu"                      # Intel Macs
```

## macOS-Specific Commands

### File System Operations
```bash
# Open directories in Finder
open .
open outputs/

# macOS file search
find . -name "*.py" -type f
mdfind -name "llama"  # Spotlight search

# Disk usage (macOS format)
df -h .
du -sh outputs/
```

### System Monitoring
```bash
# Memory usage
vm_stat
top -o MEM

# CPU monitoring  
top -o CPU
htop  # if installed via brew

# Process management
ps aux | grep python
pgrep -f llama_lora
```

### Performance Monitoring
```bash
# Activity Monitor (GUI)
open -a "Activity Monitor"

# Command line monitoring
sudo powermetrics --sample-rate 1000 -n 10  # Power/thermal

# Memory pressure
memory_pressure
```

## Apple Silicon Specific Considerations

### Memory Optimization
- **Unified Memory**: RAM and GPU memory shared
- **Memory Pressure**: Monitor system memory more carefully
- **Batch Size**: May need smaller batches due to memory architecture

### Performance Characteristics
- **CPU Performance**: Excellent for inference and small training
- **MPS Performance**: Good for medium workloads, but not CUDA-level
- **Power Efficiency**: Excellent thermal and power characteristics

### Troubleshooting MPS Issues
```bash
# Check MPS availability
python -c "
import torch
print(f'MPS available: {torch.backends.mps.is_available()}')
print(f'MPS built: {torch.backends.mps.is_built()}')
"

# MPS environment variables
export PYTORCH_ENABLE_MPS_FALLBACK=1
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
```

## Package Management (uv on macOS)

### Installation & Setup
```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Update uv
uv self update

# Project setup
uv sync
```

### Virtual Environment Management
```bash
# Check Python version
uv python list
uv python install 3.12

# Environment info
uv venv --python 3.12
source .venv/bin/activate  # if using venv manually
```

## Development Tools Integration

### VS Code Integration
- Uses system Python 3.12 with uv
- Ruff extension for linting/formatting
- Python extension for debugging

### Jupyter Integration
```bash
# Start Jupyter on macOS
uv run jupyter lab
# Automatically opens browser to localhost:8888
```

### Git Integration
```bash
# macOS git (system or Xcode Command Line Tools)
git --version

# Common workflows
git status
git add .
git commit -m "message"
```

## Hardware Compatibility

### Supported Configurations
- **M1/M2/M3 MacBooks**: Primary target, excellent performance
- **M1/M2 Mac Studio**: High-memory configurations ideal
- **Intel Macs**: CPU-only, slower but functional
- **External GPUs**: CUDA support possible with eGPU setups

### Memory Requirements
- **Minimum**: 16GB unified memory (M1/M2)
- **Recommended**: 32GB+ for larger models
- **Training**: 1B model requires ~8-12GB memory for LoRA

## Productivity Features

### Spotlight Integration
```bash
# Quick file finding
mdfind "llama_lora" -onlyin .
mdfind -name "config.yaml"
```

### Quick Look & Preview
```bash
# Preview files without opening
qlmanage -p config/config.yaml
```

### Directory Navigation
```bash
# Recent directories (if using zsh)
cd -
dirs -v

# Quick navigation
pushd outputs/
popd
```

## Networking & Downloads

### Model Downloads
```bash
# Check internet connectivity for HF downloads
ping huggingface.co
curl -I https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct

# Monitor download progress
nettop  # Network activity monitor
```

### Firewall Considerations
- **TensorBoard**: Port 6006 (usually allowed)
- **Jupyter**: Port 8888 (usually allowed)
- **HuggingFace**: HTTPS downloads (port 443)