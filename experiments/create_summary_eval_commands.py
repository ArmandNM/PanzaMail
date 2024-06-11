import json
from os import makedirs
from os.path import abspath, dirname, exists, join

CURRENT_DIR = dirname(abspath(__file__))
DATA_DIR_ROOT = abspath(join(CURRENT_DIR, "../data"))
PREAMBLES_DIR_ROOT = abspath(join(CURRENT_DIR, "../prompt_preambles"))
SCRIPTS_DIR_ROOT = abspath(join(CURRENT_DIR, "../scripts"))
SUMMARIZATION_PROMPT = abspath(
    join(CURRENT_DIR, "../src/panza/data_preparation/summarization_prompt.txt")
)
EVAL_SCRIPT = abspath(
    join(CURRENT_DIR, "../src/panza/evaluation/evaluate_summaries.py")
)


EXPERIMENT_LOG_DIR = abspath(
    join(CURRENT_DIR, "../results/summarization")
)  # Modify this accordingly
if not exists(EXPERIMENT_LOG_DIR):
    makedirs(EXPERIMENT_LOG_DIR)
COMMANDS_PATH = join(EXPERIMENT_LOG_DIR, "commands.txt")

PANZA_GENERATIVE_MODEL = "microsoft/Phi-3-mini-4k-instruct"
GENERATIVE_MODELS = [
    "ISTA-DASLab/Meta-Llama-3-8B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.2",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "microsoft/Phi-3-mini-4k-instruct",
]

USERS = [
    ("david", "/nfs/scistore19/alistgrp/anicolic/repos/PanzaMailFork/data/david_llama3/david_ground_truth_summaries.jsonl"),
]

EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
BATCH_SIZE = 8

SEEDS = [41, 42, 43, 44, 45, 46, 47, 48, 49, 50]


def generate_experiments():
    commands = []
    for model in GENERATIVE_MODELS:
        for user, golden in USERS:
            for seed in SEEDS:
                command = f"python {EVAL_SCRIPT} --model={model} --golden-loc={golden} --prompt-file={SUMMARIZATION_PROMPT} --batch-size={BATCH_SIZE} --outputs-dir={EXPERIMENT_LOG_DIR} --seed={seed}"
                commands.append(command)

    return commands


def main():
    commands = generate_experiments()

    with open(COMMANDS_PATH, "w") as f:
        for command in commands:
            f.write(command + "\n")


if __name__ == "__main__":
    main()
