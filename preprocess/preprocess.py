import argparse
import logging
import os

import pandas as pd
import wandb

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):
    run = wandb.init(job_type="preprocess_data")

    logger.info("Downloading artifact")
    artifact = run.use_artifact(args.input_artifact)
    artifact_path = artifact.file()
    df = pd.read_parquet(artifact_path)

    # Drop duplicates
    logger.info("Dropping duplicates")
    df = df.drop_duplicates().reset_index(drop=True)

    # A minimal feature engineering step: a new feature
    logger.info("Feature engineering")
    df['title'].fillna(value='', inplace=True)
    df['song_name'].fillna(value='', inplace=True)
    df['text_feature'] = df['title'] + ' ' + df['song_name']

    filename: str = "processed_data.csv"
    df.to_csv(filename)

    # save to artifact in wandb
    artifact = wandb.Artifact(
        name=args.artifact_name,
        type=args.artifact_type,
        description=args.artifact_description,
    )

    artifact.add_file(filename)
    logger.info("Logging artifact")
    run.log_artifact(artifact)

    # remove file from local env
    os.remove(filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_artifact",
        type=str,
        required=True
    )

    parser.add_argument(
        "--artifact_name",
        type=str,
        required=True
    )

    parser.add_argument(
        "--artifact_type",
        type=str,
        required=True
    )

    parser.add_argument(
        "--artifact_description",
        type=str,
        required=True
    )

    args = parser.parse_args()
    go(args)
