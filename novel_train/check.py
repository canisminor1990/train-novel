"""Check that this machine can see the 4090 from the training environment."""

from __future__ import annotations

import sys

from novel_train.envload import configure_stdio, load_project_env

load_project_env()


def main() -> int:
    configure_stdio()
    import torch
    import unsloth

    print(f"torch {torch.__version__}")
    print(f"cuda {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("PyTorch 没有看到 GPU。")
        return 1
    free, total = torch.cuda.mem_get_info()
    print(f"gpu {torch.cuda.get_device_name(0)}")
    print(f"bf16 {torch.cuda.is_bf16_supported()}")
    print(f"vram_free_mib {free / 1024 / 1024:.0f} / {total / 1024 / 1024:.0f}")
    print(f"unsloth {getattr(unsloth, '__version__', 'imported')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
