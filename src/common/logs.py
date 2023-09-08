import logging

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

    file_handler = logging.FileHandler(filename=f'./logs/{log_file}.log')
    file_handler.setFormatter(log_fmt)
    file_handler.setLevel(log_file_level)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger 
