# coding: utf-8

import argparse
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from .falpr import foo


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--image',
                        required=True,
                        help='Path to input image')
    return parser.parse_args()


def main():
    args = parse_args()
    img_fn = args.image
    if not os.path.exists(img_fn):
        print('No such file')
        return
    print(foo(img_fn))


if __name__ == '__main__':
    main()
