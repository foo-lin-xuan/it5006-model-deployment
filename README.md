## Setting Up

1. Create virtual environment with `Python 3.11.7`.
1. Activate environment.
1. Install requirements.
    
    ```
    pip install -r requirements.txt
    ```

### Virtual Environment

1. Install Miniconda and initialise conda for shell interaction.
1. Open Anaconda Prompt and run `conda init <shell>`
1. Create environment:

    ``` 
    conda create -n <my-env> python=3.11.7
    ```

1. Activate the environment:

        ```
        conda activate <my-env>
        ```

### Useful commands

| Command | Description |
| --- | --- |
| `conda init <shell>` | Initialise conda for shell interaction. <br> (E.g. `conda init cmd.exe`) |
| `conda deactivate` | Deactivate environment |
| `conda info --envs` | List environments |
| `conda env remove --name <my-env>` | Remove the environment | 

[Conda Documentation](https://docs.conda.io/projects/conda/en/stable/index.html)

## Part 1 - Model Training

```
Model
|   L6.1_Fraud_Detection_Training.ipynb     # Model Training
├───models                                  # Outputs:
│       feature_stats.json                  #       For drift detection
│       model_metadata.json                 #       Features, categories, metrics
│       random_forest_pipeline.pkl          #       Preprocessor + Random Forest
│       xgboost_pipeline.pkl                #       Preprocessor + XGBoost
│
└───models-figs
        confusion_matrices.png
```

## Part 2 - FastAPI Service

```
FastAPI_and_Render
├───Deploy_Render
│   └───models
└───Part2_FastAPI-local
    ├───logs
    ├───models
    └───__pycache__
```

### Local Testing

1. Navigate to `Part2_FastAPI-local/`
1. Run `uvicorn main:app --reload`
1. Open `localhost:8000/docs` to see Interactive Swagger UI

| Endpoints | Description |
| --- | --- |
| `http://localhost:8000/` | Health check |
| `http://localhost:8000/docs` | Interactive API docs |