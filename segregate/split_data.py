import argparse
import logging
import os
import tempfile

import pandas as pd
import wandb
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

def go(args):
    pass



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    args = parser.parse_args()
    go(args)