# -*- coding: utf-8 -*-
import click
from src.utils.click import PathlibPath
from pathlib import Path
import logging
import src.utils.joblib as ujob

import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
import eli5
from src.models.model import make_model


PROJECT_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DATA_DIR = PROJECT_DIR.joinpath('data/processed/')
DATASET_DIR = PROCESSED_DATA_DIR.joinpath('dataset')
TRAIN_TEST_DIR = PROCESSED_DATA_DIR.joinpath('train_test')
MODELS_DIR = PROJECT_DIR.joinpath('models')


def main(input_path=TRAIN_TEST_DIR, output_path=MODELS_DIR):
    output_path.mkdir(parents=True, exist_ok=True)
    X_train, X_test, y_train, y_test = ujob.load_multiple(input_path, 
        ['X_train.joblib', 'X_test.joblib', 'y_train.joblib', 'y_test.joblib'])
    
    model = make_model()
    
    cv = StratifiedKFold(n_splits=5)
    
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
    print(f'CV scores: {cv_scores}')
    print(f'Mean CV score: {np.mean(cv_scores)}')
    
    model.fit(X_train, y_train)
    
    formatter = eli5.formatters.text.format_as_text
    print(formatter(eli5.explain_weights(model)))
    
    y_test_pred = model.predict(X_test)
    y_test_pred_proba = model.predict_proba(X_test)
    errors_mask = (y_test_pred != y_test)
    errors = np.array([np.arange(X_test.shape[0]), y_test_pred, y_test]).transpose()[errors_mask]
    errors_proba = y_test_pred_proba[errors_mask]
    
    ujob.dump_multiple([model, errors, errors_proba], 
                       output_path, 
                       ['model.joblib', 'errors.joblib', 'errors_proba.joblib'])


@click.command()
@click.option('-i', 'input_path', 
              type=PathlibPath(exists=True, file_okay=False), default=TRAIN_TEST_DIR, 
              help='Input directory for the train/test dataset parts (default: <project_dir>/data/processed/train_test)')
@click.option('-o', 'output_path', 
              type=PathlibPath(exists=True, file_okay=False), default=MODELS_DIR, 
              help='Output directory for the model (default: <project_dir>/models)')
def cli(input_path, output_path):
    """ Evaluate and export the trained model.
    """
    logger = logging.getLogger(__name__)
    logger.info('evaluating the model')
    
    main(input_path, output_path)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()