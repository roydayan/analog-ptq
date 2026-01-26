#!/usr/bin/env python3
"""CLI script to run quantization experiments.

Usage:
    python scripts/run_experiment.py configs/examples/llama_gptq.yaml
    
    # With custom logging
    python scripts/run_experiment.py configs/my_config.yaml --log-level DEBUG
"""

import sys
from pathlib import Path

# Add parent directory to path for development
sys.path.insert(0, str(Path(__file__).parent.parent))

from analog_ptq.pipeline.runner import main

if __name__ == "__main__":
    sys.exit(main())
