#!/usr/bin/env python3
"""Command Line Interface (CLI) Execution Driver."""

import click

from src.preprocess_occurrences import run_preprocess
from src.data_prep import run_data_prep
from src.block_cv import run_block_cv
from src.model_training import run_model_training
from src.predict_ensemble import run_predict_ensemble
from src.compute_ibii_exports import run_compute_ibii


@click.command()
@click.option(
    "--step",
    "-s",
    default="all",
    type=click.Choice(
        ["preprocess", "prep", "block_cv", "train", "predict", "compute", "all"],
        case_sensitive=False,
    ),
    help="Pipeline step to run.",
)
def main(step):
    """Run the Insect Biodiversity Intactness Index (i-BII) pipeline."""
    if step in ["preprocess", "all"]:
        run_preprocess()
    if step in ["prep", "all"]:
        run_data_prep()
    if step in ["block_cv", "all"]:
        run_block_cv()
    if step in ["train", "all"]:
        run_model_training()
    if step in ["predict", "all"]:
        run_predict_ensemble()
    if step in ["compute", "all"]:
        run_compute_ibii()

    print("Pipeline execution finished successfully.")


if __name__ == "__main__":
    main()
