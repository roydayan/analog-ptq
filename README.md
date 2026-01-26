# Analog PTQ

A modular post-training quantization pipeline for Large Language Models (LLMs) with customizable algorithms and integrated benchmarking.

## Features

- **Modular Architecture**: Easy to extend with custom quantization methods
- **GPTQ Implementation**: Fully customizable GPTQ quantizer with overridable components
- **HuggingFace Integration**: Seamless loading of models from HuggingFace Hub
- **Benchmark Suite**: Integrated with lm-evaluation-harness for comprehensive evaluation
- **Config-Driven**: YAML-based experiment configuration
- **Extensible Registry**: Register custom quantizers, calibration loaders, and evaluators

## Installation

### From Source

```bash
git clone https://github.com/roydayan/analog-ptq.git
cd analog-ptq
pip install -e .
```

### Dependencies

```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Using Python API

```python
from analog_ptq import load_model, GPTQQuantizer, run_experiment
from analog_ptq.models import ModelWrapper, load_tokenizer
from analog_ptq.quantization import CalibrationDataLoader

# Load model
model = load_model("meta-llama/Llama-2-7b-hf", dtype="float16")
tokenizer = load_tokenizer("meta-llama/Llama-2-7b-hf")
wrapper = ModelWrapper(model)

# Prepare calibration data
calib_loader = CalibrationDataLoader(tokenizer, seq_length=2048)
calibration_data = calib_loader.load_dataset_samples("wikitext", num_samples=128)

# Quantize
quantizer = GPTQQuantizer(bits=4, group_size=128)
wrapper = quantizer.prepare(wrapper)
wrapper = quantizer.quantize(wrapper, calibration_data)

# Save
quantizer.save(wrapper, "./outputs/quantized_model")
```

### 2. Using YAML Configuration

Create a configuration file:

```yaml
# my_experiment.yaml
experiment:
  name: "llama2-7b-gptq-4bit"
  output_dir: "./outputs"

model:
  name_or_path: "meta-llama/Llama-2-7b-hf"
  dtype: "float16"
  device_map: "auto"

quantization:
  method: "gptq"
  bits: 4
  group_size: 128
  calibration:
    dataset: "wikitext"
    num_samples: 128
    seq_length: 2048

evaluation:
  tasks: ["hellaswag", "arc_easy", "arc_challenge"]
  batch_size: 8
```

Run the experiment:

```bash
analog-ptq my_experiment.yaml
# or
python scripts/run_experiment.py my_experiment.yaml
```

### 3. Using Python with Config

```python
from analog_ptq.config import load_config
from analog_ptq.pipeline import run_experiment

config = load_config("my_experiment.yaml")
results = run_experiment(config)
```

## Architecture

```
analog_ptq/
├── models/           # Model loading and wrapping
│   ├── loader.py     # HuggingFace model loading
│   └── wrapper.py    # Model wrapper with layer access hooks
├── quantization/     # Quantization methods
│   ├── base.py       # Abstract base quantizer
│   ├── gptq.py       # GPTQ implementation
│   ├── calibration.py# Calibration data handling
│   └── utils.py      # Quantization utilities
├── evaluation/       # Benchmarking
│   ├── harness.py    # lm-evaluation-harness integration
│   └── metrics.py    # Performance metrics collection
├── pipeline/         # Orchestration
│   ├── runner.py     # Experiment runner
│   └── registry.py   # Component registry
├── config/           # Configuration
│   └── schema.py     # Pydantic config schema
└── utils/            # Utilities
    └── logging.py    # Logging with rich formatting
```

## Extending the Framework

### Custom Quantizer

Create a custom quantizer by subclassing `BaseQuantizer`:

```python
from analog_ptq.quantization import BaseQuantizer
from analog_ptq.pipeline import register_quantizer

@register_quantizer("my_quantizer")
class MyQuantizer(BaseQuantizer):
    def prepare(self, model):
        # Setup model for quantization
        wrapper = ModelWrapper(model) if not isinstance(model, ModelWrapper) else model
        return wrapper
    
    def quantize(self, model, calibration_data):
        # Your quantization logic here
        for layer_idx in range(model.num_layers()):
            linear_layers = model.get_linear_layers(layer_idx)
            # Process each linear layer...
        return model
```

Use in config:

```yaml
quantization:
  method: "my_quantizer"  # Uses your registered quantizer
  bits: 4
```

### Modifying GPTQ

The GPTQ implementation is designed for easy customization. Override specific methods:

```python
from analog_ptq.quantization import GPTQQuantizer

class MyGPTQ(GPTQQuantizer):
    def _compute_hessian(self, layer_inputs, linear, device):
        # Custom Hessian computation
        H = super()._compute_hessian(layer_inputs, linear, device)
        # Add your modifications...
        return H
    
    def _quantize_weight(self, W, H):
        # Custom weight quantization
        # Modify the GPTQ algorithm here
        return Q, scales, zeros
    
    def _find_optimal_order(self, H):
        # Custom column ordering strategy
        return permutation
```

## Configuration Reference

### Experiment Section

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | string | required | Experiment name |
| `output_dir` | string | `"./outputs"` | Output directory |
| `seed` | int | `42` | Random seed |

### Model Section

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name_or_path` | string | required | Model name or path |
| `dtype` | string | `"float16"` | Model dtype |
| `device_map` | string/dict | `"auto"` | Device placement |
| `trust_remote_code` | bool | `false` | Trust remote code |

### Quantization Section

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `method` | string | `"gptq"` | Quantization method |
| `bits` | int | `4` | Bit width (2, 3, 4, 8) |
| `group_size` | int | `128` | Group size (-1 for per-channel) |
| `symmetric` | bool | `false` | Symmetric quantization |
| `damp_percent` | float | `0.01` | GPTQ dampening |
| `actorder` | bool | `false` | Activation ordering |

### Evaluation Section

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `tasks` | list | `["hellaswag"]` | Evaluation tasks |
| `batch_size` | int | `8` | Batch size |
| `num_fewshot` | int | `null` | Few-shot examples |
| `limit` | int | `null` | Sample limit per task |

## Example Configs

See `configs/examples/` for example configurations:

- `llama_gptq.yaml` - LLaMA-2-7B with GPTQ 4-bit
- `mistral_gptq.yaml` - Mistral-7B with GPTQ 4-bit
- `eval_only.yaml` - Baseline evaluation without quantization

## Supported Models

Any HuggingFace causal language model, including:

- LLaMA / LLaMA-2 / LLaMA-3
- Mistral / Mixtral
- GPT-2 / GPT-J / GPT-NeoX
- Falcon
- OPT
- Pythia

## Supported Evaluation Tasks

Via lm-evaluation-harness:

- HellaSwag
- ARC (Easy & Challenge)
- WinoGrande
- MMLU
- TruthfulQA
- And many more...

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.
