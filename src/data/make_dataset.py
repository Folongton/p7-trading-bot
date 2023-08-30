# -*- coding: utf-8 -*-
import os
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

from src.common.globals import G

def main(input_filepath='', output_filepath=''):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = G.setup_logging(logger_name=__name__,
                             console_level=logging.INFO, 
                             log_file=os.path.basename(__file__), 
                             log_file_level=logging.INFO)
    logger.info('making final data set from raw data')



if __name__ == '__main__':
    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv(), verbose=True)
    # Example:  AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY")

    main()
