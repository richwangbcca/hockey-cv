"""Compatibility entry point for the labeling application."""

from .src.app import *  # noqa: F401,F403
from .src.app import main


if __name__ == '__main__':
    main()
