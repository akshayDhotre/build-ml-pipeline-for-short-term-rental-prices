#!/usr/bin/env python
"""
Download from W&B the raw dataset and apply some basic data cleaning, exporting the result to a new artifact
"""
import os
import argparse
import logging
import wandb

import pandas as pd


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    # Download input artifact. This will also log that this script is using this
    # particular version of the artifact
    # artifact_local_path = run.use_artifact(args.input_artifact).file()

    logger.info("Artifact dowload in progress")
    artifact_path = run.use_artifact(args.input_artifact).file()
    artifact_df = pd.read_csv(artifact_path)
    logger.info("Artifact downloaded and loaded for further process")

    # remove outliers
    logger.info("Dropping outliers")
    outlier_idx = artifact_df.price.between(args.min_price, args.max_price)
    artifact_df = artifact_df[outlier_idx].copy()

    # convert string/object type data to appropriate format
    logger.info("Converting last_review to datetime values")
    artifact_df.last_review = pd.to_datetime(artifact_df.last_review)

    # dropping rows that re not in proper geolocation 
    longitude_idx = artifact_df['longitude'].between(-74.25, -73.50) & artifact_df['latitude'].between(40.5, 41.2)
    artifact_df = artifact_df[longitude_idx].copy()

    # save cleaned dataset
    logger.info("Saving the output artifact")
    file_name = "clean_sample.csv"
    artifact_df.to_csv(file_name, index=False)

    # create wandd Artifact object to log it
    artifact = wandb.Artifact(
        args.output_artifact,
        type=args.output_type,
        description=args.output_description,
    )

    artifact.add_file(file_name)

    logger.info("Logging artifact to wandb run")
    run.log_artifact(artifact)

    # remove artifact from local memory
    os.remove(file_name)

    logger.info("Artifact logged and removed from local storage")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="A very basic data cleaning")


    parser.add_argument(
        "--input_artifact", 
        type=str,
        help="Name of the input artifact",
        required=True
    )

    parser.add_argument(
        "--output_artifact", 
        type=str,
        help="Name of the output artifact",
        required=True
    )

    parser.add_argument(
        "--output_type", 
        type=str,
        help="Type of the output artifact",
        required=True
    )

    parser.add_argument(
        "--output_description", 
        type=str,
        help="Description of the output artifact",
        required=True
    )

    parser.add_argument(
        "--min_price", 
        type=float,
        help="Minimum price for cleanup of outliers",
        required=True
    )

    parser.add_argument(
        "--max_price", 
        type=float,
        help="Maximum price for cleanup of outliers",
        required=True
    )


    args = parser.parse_args()

    go(args)
