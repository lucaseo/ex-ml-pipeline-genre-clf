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
            os.path.join(root_path, "download")

        )

    # 2. Preprocess step

    # 3. Validation step

    # 4. Segregation step

    # 5. Modelling step

    # 6. Evaluation step

    pass