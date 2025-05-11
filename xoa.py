from time import sleep
from tqdm import tqdm

import os

import logging

# Configure logging
logging.basicConfig(
    filename='app.log',           # Log file name
    filemode='w',                 # Append mode ('w' to overwrite)
    level=logging.NOTSET,          # Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format='%(asctime)s | %(levelname)s | %(name)s : %(message)s',  # Log format
    encoding='utf-8'
)
logger = logging.getLogger(__name__)
logger.info(os.getpid())
        
if __name__=="__main__":
    try:
        pbar = tqdm([i for i in range(100)])
        for idx, char in enumerate(pbar):
            sleep(1)
            pbar.set_description("Processing %s" % char)
            if idx % 10 == 0 and idx > 0:
                with open('abc.txt', 'r') as file:
                    pass
            if idx % 2 == 0:
                logger.info(f"Loss: {idx}")
    except Exception as e:
        logger.error(e)