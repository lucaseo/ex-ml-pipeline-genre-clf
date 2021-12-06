import os

import mlflow
import hydra
from omegaconf import DictConfig



## Todo
## add hydra configuration
## attach wandb
## attach each step
@hydra.main(config_name="config")
def main(config):

    # Setup the wandb experiment.
    os.environ["WNADB_PROJECT"] = config["main"]["project_name"]


    # Get the path at the root of the MLflow project with this:
    # https://hydra.cc/docs/tutorials/basic/running_your_app/working_directory/
    root_path = hydra.utils.get_original_cwd()

    # Check which steps we need to execute
    if isinstance(config["main"]["execute_steps"], str):
        # This was passed on the command line as a comma-separated list of steps
        steps_to_execute = config["main"]["execute_steps"].split(",")
    else:
        steps_to_execute = config["main"]["execute_steps"]

    # Chained pipeline
    # 1. Download step
    if "download" in steps_to_execute:
        mlflow.run(
            os.path.join(root_path, "download"),
            "main",    # entrypoint to run
            parameters={
                "file_url": config["data"]["file_url"],
                "artifact_name": "raw_data.parquet",
                "artifact_type": "raw_data",
                "artifact_description": "Data as download"
            },
            use_conda=False
        )

    # 2. Preprocess step
    if "preprocess" in steps_to_execute:
        mlflow.run(
            os.path.join(root_path, "preprocess"),
            "main",
            parameters={
                "input_artifact" : "raw_data.parquet:latest",
                "artifact_name" : "preprocessed_data.csv",
                "artifact_type" : "preprocessed_data",
                "artifact_description" : "Preprocessing applied data"
            },
            use_conda=False
        )

    # 3. Validation step
    if "validate_data" in steps_to_execute:
        mlflow.run(
            os.path.join(root_path, "validate_data"),
            "main",
            parameters={
                "reference_artifact": config['data']['reference_dataset'],
                "sample_artifact": "preprocessed_data.csv:latest",
                "ks_alpha": config['data']['ks_alpha']
            }
        )

    # 4. Segregation step
    if "segregate" in steps_to_execute:
        mlflow.run(
            os.path.join(root_path, "segregate"),
            "main",
            parameters={

            }
        )

    # 5. Modelling step

    # 6. Evaluation step

    pass