import os
import yaml
from pathlib import Path

from src.loader import DataLoader
from src.solver import Solver

def main(config):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config["gpu_no"])

    save_path = Path(config["training"]["save_path"])
    save_path.mkdir(parents=True, exist_ok=True)

    mode = config["mode"]
    solver = Solver(config)

    if mode == "train":
        solver.train()
    if mode == "test":
        solver.test()


if __name__ == "__main__":
    with open("./config.yaml") as f:
        config = yaml.load(f)

    main(config)
