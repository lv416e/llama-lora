# System-Specific Notes (Darwin/macOS)

## Environment Details
- **OS**: Darwin (macOS)
- **Python**: 3.12+ (managed via uv)
- **Architecture**: Supports both Intel and Apple Silicon (M1/M2/M3)

## Hardware Considerations

### GPU/Acceleration Support
- **NVIDIA GPUs**: CUDA support for training acceleration
- **Apple Silicon**: MPS (Metal Performance Shaders) backend
- **CPU Fallback**: Automatic fallback for compatibility

### Memory Requirements
| Model Size | LoRA (fp16) | DoRA (fp16) | Full Fine-tune |
|------------|-------------|-------------|----------------|
| 1B params  | 4-6GB VRAM  | 6-8GB VRAM  | 12-16GB VRAM   |
| 3B params  | 8-12GB VRAM | 12-16GB VRAM| 24-32GB VRAM   |
| 7B params  | 16-24GB VRAM| 24-32GB VRAM| 48-64GB VRAM   |

## macOS-Specific Commands
```bash
# Open files/directories in Finder
open .
open outputs/

# Check system information
system_profiler SPHardwareDataType  # Hardware info
sysctl -n machdep.cpu.brand_string  # CPU info
```

## Common Issues on macOS

### Flash Attention Compatibility
- May not be available on older macOS versions
- Automatic fallback to standard attention implemented
- Check CUDA version compatibility if using external GPU

### Permission Issues
- Use `chmod` for permission fixes
- Some test cases may skip on macOS due to filesystem restrictions
- Recommended to use virtual environments

### Performance Tips
- Use Apple Silicon MPS when available
- Consider reducing batch size for memory-constrained systems
- Monitor Activity Monitor for resource usage during training

## Development Environment
- **Terminal**: Works with any terminal (Terminal.app, iTerm2, etc.)
- **Package Management**: uv (faster than pip on macOS)
- **File Watching**: Native fsevents support