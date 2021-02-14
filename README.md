

ml_template
==============================

Basic machine learning project template based on <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science</a>, using iris dataset as an example. The goal is to outline and structurize the most necessary routines of a machine learning project so that one could focus more on the machine learning itself.

Features
------------
- A complete pipeline, from getting raw data to storing the trained model and evaluation metrics, separated into steps
- Command-line interface for running the steps as well as the whole pipeline
- Model in a form of sklearn pipeline, described in a separate file for easy experimenting
- Feature importance using ELI5
- Default dockerfile and scripts for building a container suitable both for batch and online inference
- Online inference via RESTful API 

Quickstart
------------

Use `python3 main.py run-all` to run the pipeline. This produces a serialized model file, `models/model.joblib`.

Build a Docker image: 
`docker build -t ml-template -f Dockerfile .`

To run a container in batch inference mode: 
`docker run --rm ml-template python3 batch_inference.py`

To run a container in online inference mode: 
`docker run --rm -it -p 5000:5000 ml-template python3 api.py`

An example of getting predictions in online inference mode is provided in `notebooks/api_test.ipynb`.

Usage
------------

Use `python3 main.py` for help.
