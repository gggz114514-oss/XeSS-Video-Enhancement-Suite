#!/usr/bin/env python3
"""Independent FG 1.2 motion/depth preparation.

This entrypoint never reads SR motion/depth caches; it always analyses the
finished-resolution color frames that XeSS-FG will consume.
"""

import argparse

from prepare_common import add_common_arguments, run_preparer


def main():
    parser = argparse.ArgumentParser(description="Prepare independent XeSS FG 1.2 frame data")
    add_common_arguments(parser, kind="fg")
    parser.set_defaults(responsive_mask=False, mv_path="lowres-depth")
    run_preparer(parser.parse_args())


if __name__ == "__main__":
    main()
