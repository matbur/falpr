#!/usr/bin/env python

import os
from multiprocessing.pool import Pool
from time import time

import requests

from logger import get_logger

logger = get_logger('scraper')


def scrap(url, bytes_=False):
    try:
        req = requests.get(url)
        req.raise_for_status()
    except (requests.exceptions.ConnectionError, requests.exceptions.HTTPError) as err:
        logger.error(err)
        return

    if bytes_:
        return req.content

    return req.text


def save(fn, data, bytes_=False):
    mode = ('w', 'wb')[bytes_]
    with open(fn, mode) as f:
        f.write(data)


def worker(data):
    fn, url, bytes_ = data
    data = scrap(url, bytes_)

    if data is None:
        return

    save(fn, data, bytes_)
    logger.info(url)


def main():
    t = time()

    data_dir = os.path.join('..', 'data')
    gallery_dir = os.path.join(data_dir, 'gallery')
    url = 'http://platesmania.com/pl/gallery'

    gallery_range = range(5762)
    fns = [os.path.join(gallery_dir, f'gallery-{i}.html') for i in gallery_range]
    urls = [f'{url}-{i}' for i in gallery_range]
    data = zip(fns, urls)

    with Pool(4) as pool:
        pool.map(worker, data)

    logger.info(f'Done in {time() - t}')


if __name__ == '__main__':
    main()
