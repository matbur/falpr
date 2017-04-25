#!/usr/bin/env python
import logging


def get_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(f'{name}.log')
    file_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    stream_hander = logging.StreamHandler()
    stream_hander.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_hander)

    return logger
