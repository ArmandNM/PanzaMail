import json
from os import makedirs
from os.path import abspath, dirname, exists, join

CURRENT_DIR = dirname(abspath(__file__))
DATA_DIR_ROOT = abspath(join(CURRENT_DIR, "../data"))
PREAMBLES_DIR_ROOT = abspath(join(CURRENT_DIR, "../prompt_preambles"))
SCRIPTS_DIR_ROOT = abspath(join(CURRENT_DIR, "../scripts"))

SWEEP_CONFIG_FILE = join(CURRENT_DIR, "sweep_config.json")  # Modify this accordingly

EXPERIMENT_LOG_DIR = join(CURRENT_DIR, "logs_all_users")  # Modify this accordingly
if not exists(EXPERIMENT_LOG_DIR):
    makedirs(EXPERIMENT_LOG_DIR)
COMMANDS_PATH = join(EXPERIMENT_LOG_DIR, "commands.txt")


def generate_experiments(sweep_config):
    commands = []
    for user_json in sweep_config:
        username = user_json["user_name"]
        preamble_path = join(
            PREAMBLES_DIR_ROOT, f"{user_json['user_nick']}_preamble.txt"
        )
        panza_data_dir = join(DATA_DIR_ROOT, user_json["data_dir"])
        data_path = join(panza_data_dir, "train_raft.jsonl")
        for seed in user_json["SEED"]:
            for preamble in user_json["PREAMBLE"]:
                for raft in user_json["RAFT"]:
                    if raft and not preamble:
                        continue  # Only use RAG along with system and user preambles
                    for lr in user_json["FFT_LR"]:
                        command = f"./train_fft.sh PANZA_DATA_DIR={panza_data_dir} PANZA_USERNAME={username} DATA_PATH={data_path} LR={lr} PANZA_FINETUNE_WITH_PREAMBLE={preamble} PANZA_USER_PREAMBLE_PATH={preamble_path} PANZA_FINETUNE_WITH_RAG={raft} PANZA_SEED={seed}"
                        command = f"cd {SCRIPTS_DIR_ROOT} && {command}"
                        commands.append(command)
                    for epochs in user_json["ROSA_EPOCHS"]:
                        for lr in user_json["ROSA_LR"]:
                            command = f"./train_rosa.sh PANZA_DATA_DIR={panza_data_dir} PANZA_USERNAME={username} DATA_PATH={data_path} LR={lr} LORA_LR={lr} NUM_EPOCHS={epochs} PANZA_FINETUNE_WITH_PREAMBLE={preamble} PANZA_USER_PREAMBLE_PATH={preamble_path} PANZA_FINETUNE_WITH_RAG={raft} PANZA_SEED={seed}"
                            command = f"cd {SCRIPTS_DIR_ROOT} && {command}"
                            commands.append(command)
                    for epochs in user_json["LORA_EPOCHS"]:
                        for lr in user_json["LORA_LR"]:
                            command = f"./train_rosa.sh PANZA_DATA_DIR={panza_data_dir} PANZA_USERNAME={username} DATA_PATH={data_path} LR=0.0 LORA_LR={lr} NUM_EPOCHS={epochs} PANZA_FINETUNE_WITH_PREAMBLE={preamble} PANZA_USER_PREAMBLE_PATH={preamble_path} PANZA_FINETUNE_WITH_RAG={raft} PANZA_SEED={seed}"
                            command = f"cd {SCRIPTS_DIR_ROOT} && {command}"
                            commands.append(command)

    return commands

def main():
    with open(SWEEP_CONFIG_FILE, "r") as f:
        sweep_config = json.load(f)

    commands = generate_experiments(sweep_config)

    with open(COMMANDS_PATH, "w") as f:
        for command in commands:
            f.write(command + "\n")


if __name__ == "__main__":
    main()
