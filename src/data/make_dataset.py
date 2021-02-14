# -*- coding: utf-8 -*-
import click
from src.utils.click import PathlibPath
from pathlib import Path
import logging
import src.utils.joblib as ujob

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_DIR.joinpath('data/raw/')
PROCESSED_DATA_DIR = PROJECT_DIR.joinpath('data/processed/')
DATASET_DIR = PROCESSED_DATA_DIR.joinpath('dataset')


def main(input_path=RAW_DATA_DIR, output_path=DATASET_DIR):
    output_path.mkdir(parents=True, exist_ok=True)
    iris = ujob.load(input_path, 'iris.joblib')
    
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = iris.target
    y_labels = dict(zip( range(len(iris.target_names)), iris.target_names ))
    
    ujob.dump_multiple([X, y, y_labels], 
                       output_path, 
                       ['X.joblib', 'y.joblib', 'y_labels.joblib'])


@click.command()
@click.option('-i', 'input_path', 
              type=PathlibPath(exists=True, file_okay=False), default=RAW_DATA_DIR, 
              help='Input directory for raw data files (default: <project_dir>/data/raw)')
@click.option('-o', 'output_path', 
              type=PathlibPath(exists=True, file_okay=False), default=DATASET_DIR, 
              help='Output directory for the dataset (default: <project_dir>/data/processed/dataset)')
def cli(input_path, output_path):
    """ Process raw data to construct a dataset suitable for model pipeline input.
    """
    logger = logging.getLogger(__name__)
    logger.info('making a dataset from raw data')
    
    main(input_path, output_path)
        

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
