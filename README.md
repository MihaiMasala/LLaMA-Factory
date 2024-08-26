# Training of RoLLMs

Official code used for training Romanian LLMs as proposed in [Masala et al. 2024](https://arxiv.org/abs/2406.18266). This repo is a fork of the popular [hiyouga/LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) repo used for training and finetuning LLMs. On top of the existing framework we added a suite of Romanian datasets:

- [ro_sft_alpaca](https://huggingface.co/datasets/OpenLLM-Ro/ro_sft_alpaca)
- [ro_sft_alpaca_gpt4](https://huggingface.co/datasets/OpenLLM-Ro/ro_sft_alpaca_gpt4)
- [ro_sft_dolly](https://huggingface.co/datasets/OpenLLM-Ro/ro_sft_dolly)
- [ro_sft_selfinstruct_gpt4](https://huggingface.co/datasets/OpenLLM-Ro/ro_sft_selfinstruct_gpt4)
- [ro_sft_norobts](https://huggingface.co/datasets/OpenLLM-Ro/ro_sft_norobots)
- [ro_sft_orca](https://huggingface.co/datasets/OpenLLM-Ro/ro_sft_orca)
- [ro_sft_camel](https://huggingface.co/datasets/OpenLLM-Ro/ro_sft_camel)


![# LLaMA Factory](assets/logo.png)



- [Requirement](#requirement)
- [Getting Started](#getting-started)


## Requirement

| Mandatory    | Minimum | Recommend |
| ------------ | ------- | --------- |
| python       | 3.8     | 3.11      |
| torch        | 1.13.1  | 2.4.0     |
| transformers | 4.41.2  | 4.43.4    |
| datasets     | 2.16.0  | 2.20.0    |
| accelerate   | 0.30.1  | 0.32.0    |
| peft         | 0.11.1  | 0.12.0    |
| trl          | 0.8.6   | 0.9.6     |

| Optional     | Minimum | Recommend |
| ------------ | ------- | --------- |
| CUDA         | 11.6    | 12.2      |
| deepspeed    | 0.10.0  | 0.14.0    |
| bitsandbytes | 0.39.0  | 0.43.1    |
| vllm         | 0.4.3   | 0.5.0     |
| flash-attn   | 2.3.0   | 2.6.3     |

### Hardware Requirement

\* *estimated*

| Method            | Bits |   7B  |  13B  |  30B  |   70B  |  110B  |  8x7B |  8x22B |
| ----------------- | ---- | ----- | ----- | ----- | ------ | ------ | ----- | ------ |
| Full              | AMP  | 120GB | 240GB | 600GB | 1200GB | 2000GB | 900GB | 2400GB |
| Full              |  16  |  60GB | 120GB | 300GB |  600GB |  900GB | 400GB | 1200GB |
| Freeze            |  16  |  20GB |  40GB |  80GB |  200GB |  360GB | 160GB |  400GB |
| LoRA/GaLore/BAdam |  16  |  16GB |  32GB |  64GB |  160GB |  240GB | 120GB |  320GB |
| QLoRA             |   8  |  10GB |  20GB |  40GB |   80GB |  140GB |  60GB |  160GB |
| QLoRA             |   4  |   6GB |  12GB |  24GB |   48GB |   72GB |  30GB |   96GB |
| QLoRA             |   2  |   4GB |   8GB |  16GB |   24GB |   48GB |  18GB |   48GB |

## Getting Started

### Installation

> [!IMPORTANT]
> Installation is mandatory.

```bash
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]"
```

Extra dependencies available: torch, torch-npu, metrics, deepspeed, bitsandbytes, hqq, eetq, gptq, awq, aqlm, vllm, galore, badam, adam-mini, qwen, modelscope, quality

> [!TIP]
> Use `pip install --no-deps -e .` to resolve package conflicts.

### Data Preparation

Please refer to [data/README.md](data/README.md) for checking the details about the format of dataset files. You can either use datasets on HuggingFace / ModelScope hub or load the dataset in local disk.

> [!NOTE]
> Please update `data/dataset_info.json` to use your custom dataset.

### Quickstart

Use the following 3 commands to run LoRA **fine-tuning**, **inference** and **merging** of the Llama3-8B-Instruct model, respectively.

```bash
llamafactory-cli train examples/train_lora/llama3_lora_sft.yaml
llamafactory-cli chat examples/inference/llama3_lora_sft.yaml
llamafactory-cli export examples/merge_lora/llama3_lora_sft.yaml
```

See [examples/README.md](examples/README.md) for advanced usage (including distributed training).

> [!TIP]
> Use `llamafactory-cli help` to show help information.

### Use W&B Logger

To use [Weights & Biases](https://wandb.ai) for logging experimental results, you need to add the following arguments to yaml files.

```yaml
report_to: wandb
run_name: test_run # optional
```

Set `WANDB_API_KEY` to [your key](https://wandb.ai/authorize) when launching training tasks to log in with your W&B account.


## Cite as


```bibtex
@misc{masala2024vorbecstiromanecsterecipetrain,
      title={"Vorbe\c{s}ti Rom\^ane\c{s}te?" A Recipe to Train Powerful Romanian LLMs with English Instructions}, 
      author={Mihai Masala and Denis C. Ilie-Ablachim and Alexandru Dima and Dragos Corlatescu and Miruna Zavelca and Ovio Olaru and Simina Terian and Andrei Terian and Marius Leordeanu and Horia Velicu and Marius Popescu and Mihai Dascalu and Traian Rebedea},
      year={2024},
      eprint={2406.18266},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2406.18266}, 
}
```

### Acknowledgement
This repo benefits from [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory). We thank them for their wonderful work.


