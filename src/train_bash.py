from llmtuner import run_exp
import os
os.environ["WANDB_PROJECT"] = "llama_factory"


def main():
    run_exp()


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
