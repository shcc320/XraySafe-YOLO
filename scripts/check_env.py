from __future__ import annotations

import platform
import sys

print('Python:', sys.version)
print('Platform:', platform.platform())
try:
    import torch
    print('PyTorch:', torch.__version__)
    print('CUDA available:', torch.cuda.is_available())
    if torch.cuda.is_available():
        print('CUDA device count:', torch.cuda.device_count())
        for i in range(torch.cuda.device_count()):
            print(f'GPU {i}:', torch.cuda.get_device_name(i))
except Exception as e:
    print('PyTorch check failed:', e)
try:
    import ultralytics
    print('Ultralytics:', ultralytics.__version__)
except Exception as e:
    print('Ultralytics check failed:', e)
