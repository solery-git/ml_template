import click
from pathlib import Path

from src.data import get_data, make_dataset, split_train_test
from src.models import evaluate
from src.utils.click import OrderedGroup


STAGES = [
    ('get-data', get_data), 
    ('make-dataset', make_dataset), 
    ('split-train-test', split_train_test), 
    ('evaluate', evaluate)
]


@click.group(cls=OrderedGroup, 
             help='''Pipeline entrypoint. Use 'run-all' command to run the whole pipeline
             or use the respective commands to run specific steps.\n 
             For a detailed help on a specific command, use '<command> --help'.''')
def cli():
    pass


@cli.command(help='Run all the steps (with default parameters).')
def run_all():
    for stage, script in STAGES:
        print(f'Running stage: {stage}')
        script.main()

for stage, script in STAGES:
    cli.add_command(script.cli, name=stage)


if __name__ == '__main__':
    cli()
