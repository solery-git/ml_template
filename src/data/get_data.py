# -*- coding: utf-8 -*-
import click
from src.utils.click import PathlibPath
from pathlib import Path
import logging
import src.utils.joblib as ujob

from sklearn.datasets import load_iris


PROJECT_DIR = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_DIR.joinpath('data/raw/')


def main(output_path=RAW_DATA_DIR, force=False):
    output_path.mkdir(parents=True, exist_ok=True)
    iris_path = output_path.joinpath('iris.joblib')
    
    if not iris_path.is_file() or force:
        iris = load_iris()
        ujob.dump(iris, iris_path)


@click.command()
@click.option('-o', 'output_path', 
              type=PathlibPath(exists=True, file_okay=False), default=RAW_DATA_DIR, 
              help='Output directory for raw data files (default: <project_dir>/data/raw)')
@click.option('-f', '--force', 
              is_flag=True, 
              help='Forcefully overwrite output files')
def cli(output_path, force):
    """ Obtain raw data.
    """
    logger = logging.getLogger(__name__)
    logger.info('getting raw data')
    
    main(output_path, force)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()