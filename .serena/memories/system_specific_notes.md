# System-Specific Notes (Darwin/macOS)

## Platform Information
- **OS**: Darwin (macOS)
- **Architecture**: Likely Apple Silicon (M1/M2/M3) or Intel
- **Python**: 3.12+ via UV package manager

## macOS-Specific Considerations

### GPU/Device Support
- **Apple Silicon**: Uses MPS (Metal Performance Shaders) backend
- **Intel Mac**: CPU-only unless eGPU with NVIDIA card
- **CUDA**: Not natively supported on macOS
- **Flash Attention**: Will automatically fallback to standard attention

### File System
- **Case Sensitivity**: Usually case-insensitive (check with `diskutil info /`)
- **Path Separators**: Use forward slashes (/)
- **Home Directory**: `~` or `/Users/<username>`
- **Temp Directory**: `/tmp` or `/var/folders/...`

### Common macOS Commands
```bash
# Open file/folder in Finder
open .
open file.txt

# Copy to clipboard
echo "text" | pbcopy

# Paste from clipboard
pbpaste

# Check system info
system_profiler SPSoftwareDataType
sysctl -a | grep machdep.cpu

# Monitor processes
top -o cpu
Activity Monitor (GUI)

# Check available memory
vm_stat

# Network ports
lsof -i :8080
netstat -an | grep LISTEN
```

### Development Tools
- **Xcode Command Line Tools**: May be required for some packages
  ```bash
  xcode-select --install
  ```

### Python/UV on macOS
- UV handles Python versions automatically
- Virtual environments in `.venv` directory
- No need for pyenv or conda

### Performance Considerations
- **MPS Backend**: Slower than CUDA but works for development
- **Memory**: Unified memory architecture on Apple Silicon
- **Batch Size**: May need reduction compared to NVIDIA GPUs
- **Training Speed**: Expect slower training on MPS vs CUDA

### Environment Variables
```bash
# Set in ~/.zshrc (default shell on modern macOS)
export VARIABLE_NAME="value"

# Temporary (current session only)
export HF_TOKEN="your_token"

# Check current environment
env | grep HF
```

### Known Issues on macOS
1. **Flash Attention**: Not supported, will auto-fallback
2. **CUDA Dependencies**: Will be skipped/ignored
3. **Large Models**: May hit memory limits faster
4. **File Descriptors**: May need to increase limit
   ```bash
   ulimit -n 2048
   ```

### Debugging on macOS
```bash
# Check if running on Apple Silicon
uname -m  # arm64 for Apple Silicon, x86_64 for Intel

# Python architecture
python -c "import platform; print(platform.machine())"

# Check MPS availability
python -c "import torch; print(torch.backends.mps.is_available())"

# Monitor GPU usage (Apple Silicon)
sudo powermetrics --samplers gpu_power -i 1000
```

### Package Management
- **Homebrew**: Common for system packages
  ```bash
  brew install ripgrep
  brew install watch
  ```
- **UV**: Handles Python packages and environments

### File Permissions
- Standard Unix permissions apply
- May need to grant Terminal/IDE full disk access in System Preferences

### Networking
- Firewall may block ports (check System Preferences > Security & Privacy)
- Local development usually on localhost/127.0.0.1

### Tips for macOS Development
1. Use `caffeinate` to prevent sleep during long training
2. Increase terminal buffer for long outputs
3. Use `say "training complete"` for audio notifications
4. Consider using tmux or screen for long-running processes
5. Check Activity Monitor for memory pressure

### Resource Limits
```bash
# Check current limits
ulimit -a

# Increase for development if needed
ulimit -n 4096  # file descriptors
ulimit -s 65532 # stack size
```