#!/usr/bin/env python3
"""Independent SR 1.2 motion/depth/responsive-mask preparation."""

import argparse

from prepare_common import add_common_arguments, run_preparer


def main():
    parser = argparse.ArgumentParser(description="Prepare independent XeSS SR 1.2 frame data")
    add_common_arguments(parser, kind="sr")
    run_preparer(parser.parse_args())


if __name__ == "__main__":
    main()
