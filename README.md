# End-to-end ML pipeline using MLflow 

Reference: Udacity Machine Learning DevOps Engineer Nanodegree Course

## Pipeline description

### 1. Chain each module

### 2. Train the model through pipeline

```shell
mlflow run --no-conda .
```

### 3. Production Run

- Do a production run by changing the project_name to genre_classification_prod.
    ```shell
    mlflow run --no-conda . -P hydra_options="main.project_name='genre_classification_prod'"
    ```
- Override parameter to only execute one or more steps.
- Tag the model as `prod`

### 4. Release pipeline for production

- fix the config for `prod` 
- commit push to repository
- release the repository

### 5. Run pipeline in MLflow env

- `mlflow run -v 1.0.0 [URL of your Github repo]`
- Q. Is this work in private repository? Need to try it out.


### 6. Inference

1. fetch model from wandb
    ```bash
    wandb artifact get genre_classification_prod/model_export:prod --root model
    ```

2. Deploy inference server
   - in batch
     - prepare test data
       ```bash
       wandb artifact get genre_classification_prod/data_test.csv:latest
       ```

     - run model on test dataset
       ```bash
       mlflow models predict --no-conda \
                     -t csv \
                     -i ./artifacts/data_test.csv:v0/data_test.csv \
                     -m model
       ```

   - in online
     - set up REST API
       ```bash
       mlflow models serve --no-conda -m model &
       ```


## Reference

- [How to use pytest (KR)](https://jangseongwoo.github.io/test/pytest_basic/)