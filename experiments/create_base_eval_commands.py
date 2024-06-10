import json
from os import makedirs
from os.path import abspath, dirname, exists, join

CURRENT_DIR = dirname(abspath(__file__))
DATA_DIR_ROOT = abspath(join(CURRENT_DIR, "../data"))
PREAMBLES_DIR_ROOT = abspath(join(CURRENT_DIR, "../prompt_preambles"))
SCRIPTS_DIR_ROOT = abspath(join(CURRENT_DIR, "../scripts"))
EVAL_SCRIPT = abspath(
    join(CURRENT_DIR, "../src/panza/evaluation/evaluate_bleu_score.py")
)


EXPERIMENT_LOG_DIR = abspath(join(CURRENT_DIR, "../results/logs_base_models"))  # Modify this accordingly
if not exists(EXPERIMENT_LOG_DIR):
    makedirs(EXPERIMENT_LOG_DIR)
COMMANDS_PATH = join(EXPERIMENT_LOG_DIR, "commands.txt")

SWEEP_CONFIG_FILE = join(CURRENT_DIR, "sweep_config.json")  # Modify this accordingly

PANZA_GENERATIVE_MODEL = "microsoft/Phi-3-mini-4k-instruct"
GENERATIVE_MODELS = [
    "ISTA-DASLab/Meta-Llama-3-8B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.2",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "microsoft/Phi-3-mini-4k-instruct",
]

EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
BATCH_SIZE = 8

RAG_RELEVANCE_THRESHOLD = 0.2
RAG_NUM_EMAILS = 3

SEEDS = [41, 42, 43]
# SEEDS = [41, 42, 43, 44, 45, 46, 47, 48, 49, 50]


def generate_experiments(sweep_config):
    commands = []
    commands_rag = []
    for model in GENERATIVE_MODELS:
        for user_json in sweep_config:
            username = user_json["user_name"]
            user_preamble_path = join(
                PREAMBLES_DIR_ROOT, f"{user_json['user_nick']}_preamble.txt"
            )
            system_preamble_path = join(PREAMBLES_DIR_ROOT, f"system_preamble.txt")
            rag_preamble_path = join(PREAMBLES_DIR_ROOT, f"rag_preamble.txt")
            panza_data_dir = join(DATA_DIR_ROOT, user_json["data_dir"])
            golden_data_path = join(panza_data_dir, "test.jsonl")
            for seed in SEEDS:
                command = f"python {EVAL_SCRIPT} --model={model} --system-preamble={system_preamble_path} --user-preamble={user_preamble_path} --rag-preamble={rag_preamble_path} --golden={golden_data_path} --batch-size={BATCH_SIZE} --outputs-dir={EXPERIMENT_LOG_DIR} --seed={seed}"
                command_rag = f"python {EVAL_SCRIPT} --model={model} --system-preamble={system_preamble_path} --user-preamble={user_preamble_path} --rag-preamble={rag_preamble_path} --golden={golden_data_path} --batch-size={BATCH_SIZE} --embedding-model={EMBEDDING_MODEL} --db-path={panza_data_dir} --index-name={username} --rag-relevance-threshold={RAG_RELEVANCE_THRESHOLD} --rag-num-emails={RAG_NUM_EMAILS} --use-rag --outputs-dir={EXPERIMENT_LOG_DIR} --seed={seed}"
                commands.append(command)
                commands_rag.append(command_rag)

    return commands + commands_rag


def main():
    with open(SWEEP_CONFIG_FILE, "r") as f:
        sweep_config = json.load(f)

    commands = generate_experiments(sweep_config)

    with open(COMMANDS_PATH, "w") as f:
        for command in commands:
            f.write(command + "\n")


if __name__ == "__main__":
    main()
