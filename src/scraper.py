#!/usr/bin/env python

import json
import os
from multiprocessing.pool import ThreadPool as Pool
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
    fn, url, *bytes_ = data
    bytes_ = bytes_ is not None

    data = scrap(url, bytes_)

    if data is None:
        return

    save(fn, data, bytes_)
    logger.info(url)


def platesmania(gallery_dir):
    url = 'http://platesmania.com/pl/gallery'
    gallery_range = range(5762)
    fns = [os.path.join(gallery_dir, f'gallery-{i}.html') for i in gallery_range]
    urls = [f'{url}-{i}' for i in gallery_range]
    return zip(fns, urls)


def rejestracja_blog(gallery_dir):
    url = 'http://rejestracja.blog.pl/page/{}/'
    gallery_range = range(800)
    fns = [os.path.join(gallery_dir, f'gallery-{i}.html') for i in gallery_range]
    urls = [url.format(i) for i in gallery_range]
    return zip(fns, urls)


def main():
    t = time()

    data_dir = os.path.join('..', 'data')
    images_dir = os.path.join(data_dir, 'images')

    with open('images.json') as f:
        data = list(list(i) for i in json.load(f).items())

    for i in data:
        i[0] = os.path.join(images_dir, f'{i[0]}.jpg')
        i.append(True)

    with Pool() as pool:
        pool.map(worker, data)

    logger.info(f'Done in {time() - t}')


if __name__ == '__main__':
    main()
