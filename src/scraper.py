#!/usr/bin/env python

import os
from multiprocessing.pool import Pool

import requests

from src.logger import get_logger

logger = get_logger('scrapper')


def scrap(url):
    try:
        req = requests.get(url)
    except requests.exceptions.ConnectionError:
        return
    if req.raise_for_status():
        return
    return req.text


def collect(fn, data):
    with open(fn, 'w') as f:
        f.write(data)


def worker(data):
    fn, url = data
    data = scrap(url)

    if data is None:
        logger.error('error with %s', url)
        return

    collect(fn, data)
    logger.info(url)


def main():
    data_dir = os.path.join('..', 'data')
    gallery_dir = os.path.join(data_dir, 'gallery')
    url = 'http://platesmania.com/pl/gallery'

    gallery_range = range(5762)
    fns = [os.path.join(gallery_dir, f'gallery-{i}.html') for i in gallery_range]
    urls = [f'{url}-{i}' for i in gallery_range]
    data = zip(fns, urls)

    with Pool(4) as pool:
        pool.map(worker, data)

    logger.info('done')


if __name__ == '__main__':
    main()
