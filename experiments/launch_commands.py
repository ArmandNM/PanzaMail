import queue
import subprocess
import threading
import time
from os.path import abspath, dirname, join

AVAILABLE_GPUS = [0, 1, 2, 3]

# Queue to manage GPU availability
gpu_queue = queue.Queue()
for gpu_id in AVAILABLE_GPUS:
    gpu_queue.put(gpu_id)  # Put all GPU indices into the queue

CURRENT_DIR = dirname(abspath(__file__))

EXPERIMENT_LOG_DIR = join(CURRENT_DIR, "logs")  # Modify this accordingly
COMMANDS_PATH = join(EXPERIMENT_LOG_DIR, "commands.txt")

SLEEP_BETWEEN_COMMANDS = 30  # Sleep for a few seconds before starting each subprocess

CONDA_PATH = "/nfs/scistore19/alistgrp/anicolic/anaconda3/etc/profile.d/conda.sh"
CONDA_ENV = "panza-public"
ACTIVATE_CONDA = True

def run_command(gpu_ids, command_id, command):
    gpu_ids_str = ",".join(gpu_ids)
    env = {"CUDA_VISIBLE_DEVICES": gpu_ids_str}
    log_file = join(EXPERIMENT_LOG_DIR, f"run_{command_id}.log")
    print(f"Running command {command_id}: {command}")
    print(f"ENV: {env}")
    with open(log_file, "w") as f:
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=f,
            stderr=subprocess.STDOUT,
            env=env,
            executable="/bin/bash",
        )
        process.wait()
    for gpu_id in gpu_ids:
        gpu_queue.put(gpu_id)  # Release GPU after the process completes


def manage_experiments(experiment_commands):
    while experiment_commands:
        command_id, command = experiment_commands.pop(0)
        if "train_fft" in command:
            gpu_id1 = str(gpu_queue.get())
            gpu_id2 = str(gpu_queue.get())
            gpu_ids = [gpu_id1, gpu_id2]
        else:
            gpu_ids = [str(gpu_queue.get())]  # This will block if no GPUs are available
        if command:
            print(
                "Wait for a few seconds before starting the subprocess to avoid concurrent tokenizer loading"
            )
            time.sleep(SLEEP_BETWEEN_COMMANDS)

            if ACTIVATE_CONDA:
                command = f"source {CONDA_PATH} && conda activate {CONDA_ENV} && {command}"

            threading.Thread(
                target=run_command, args=(gpu_ids, command_id, command)
            ).start()


def main():
    with open(COMMANDS_PATH, "r") as f:
        commands = f.readlines()

    commands = list(enumerate(commands))
    for experiment_id, command in commands:
        print(experiment_id)
        print(command)

    manage_experiments(commands)


if __name__ == "__main__":
    main()
