"""Run the proxy: ``PYTHONPATH=/packages python3 -m vv_longform``."""

from . import config, server

if __name__ == "__main__":
    server.main(config.load())
