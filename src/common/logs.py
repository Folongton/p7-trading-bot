import logging
import os

from env import Env

def setup_logging(console_level=logging.INFO, log_file='Main', log_file_level=logging.INFO, logger_name=__name__):
    ''' Setup Logging in log file and console
        Creates a logger object with appropriate formatting and handlers.
    INPUT: console_level: logging level for console, 
                log_file: name of log file, 
        log_file_level: logging level for log file
    OUTPUT: logger object
    '''
    log_fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_fmt)
    console_handler.setLevel(console_level)

    file_handler = logging.FileHandler(filename=f'{Env.PROJECT_ROOT}/logs/{log_file}.log') 
    file_handler.setFormatter(log_fmt)
    file_handler.setLevel(log_file_level)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger 

def log_model_info(config, model, logger):
    ''' Logs model info to console and log file, modifies model name with number of parameters
    INPUT: config: config file,
            model: model object,
            logger: logger object
    OUTPUT: model object
    '''
    # Building a header
    logger.info('='*94)
    logger.info(f"{'='*35} MODEL CONFIG AND SETUP {'='*35}")
    logger.info('='*94)

    # Getting info from Config
    for key, value in config.items():
        if key =='AV':
            av_string = ''
            for key, value in config['AV'].items():
                av_string += f'AV_{key}: {value}, '
            logger.info(av_string)
        if key == 'data':
            data_string = ''
            for key, value in config['data'].items():
                data_string += f'data_{key}: {value}, '
            logger.info(data_string)

    logger.info('-'*70)

    for key, value in config.items():
        if key == 'model':
            for key, value in config['model'].items():
                logger.info(f'model_{key}: {value}')

    model.summary(print_fn=logger.info)

