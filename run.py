import torch
from src.config import yamlparser
from src.dimslam import DIMSLAM

def main():
    args = yamlparser().args
    # print(args)

    dimslam = DIMSLAM(args)
    dimslam.start()

if __name__ == "__main__":
    main()