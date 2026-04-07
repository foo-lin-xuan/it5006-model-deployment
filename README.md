# IT5006 Fundamentals of Data Analytics | Week 6 | Introduction to Model Deployment

This project is an organisation of course materials from IT5006 Fundamentals of Data Analytics [2520].

Credits belong to the lecturer, Prakash Chandra Sukhwal.

## 📑 Table of Contents
* [Setting Up](#️-setting-up)
    * [Virtual Environment](#virtual-environment)
    * [Useful Commands](#useful-commands)
* [Part 1 - Model Training](#1️⃣-part-1---model-training)
* [Part 2 - FastAPI Service](#2️⃣-part-2---fastapi-service)
    * [Local Testing](#local-testing)  
* [Part 3 - Web Service Deployment](#3️⃣-part-3---web-service-deployment)
* [Part 4 - Application Deployment](#4️⃣-part-4---application-deployment)


## 🛠️ Setting Up

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

[↑ Back to Top](#-table-of-contents)

## 1️⃣ Part 1 - Model Training

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

[↑ Back to Top](#-table-of-contents)

## 2️⃣ Part 2 - FastAPI Service

```
FastAPI_and_Render
├───Deploy_Render
│   └───models
└───Part2_FastAPI-local     # FastAPI Service
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

[↑ Back to Top](#-table-of-contents)

## 3️⃣ Part 3 - Web Service Deployment

```
FastAPI_and_Render
├───Deploy_Render           # Web Service Deployment
│   └───models
└───Part2_FastAPI-local     
    ├───logs
    ├───models
    └───__pycache__
```

Deploy FastAPI service on [Render](https://render.com/):
- **Root Directory:** `FastAPI_and_Render/Deploy_Render`
- **Start Command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`

[↑ Back to Top](#-table-of-contents)

## 4️⃣ Part 4 - Application Deployment

```
Streamlit
└───Deploy_Streamlit        # Application Deployment
```

Deploy Streamlit application on [Streamlit Community Cloud](https://share.streamlit.io/).

[↑ Back to Top](#-table-of-contents)