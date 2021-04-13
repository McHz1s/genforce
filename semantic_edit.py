from runners.sefa_runner import SefaRunner
import json
import os
import datetime
import argparse
from utils.misc import parse_config
from utils.logger import build_logger
import shutil


def parse_args():
    """Parses arguments."""
    parser = argparse.ArgumentParser(description='Conduct Sefa.')
    parser.add_argument('--config', type=str, help='Path to the Sefa configuration.')
    return parser.parse_args()


def main():
    args = parse_args()
    # Parse configurations.
    config = parse_config(args.config)
    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpus
    timestamp = datetime.datetime.now()
    version = '%d-%d-%d-%02.0d-%02.0d-%02.0d' % \
              (timestamp.year, timestamp.month, timestamp.day, timestamp.hour, timestamp.minute, timestamp.second)
    config.work_dir = os.path.join(config.work_dir, config.checkpoint_path.split('/')[-3], version)
    logger_type = config.get('logger_type', 'normal')
    logger = build_logger(logger_type, work_dir=config.work_dir)
    shutil.copy(args.config, os.path.join(config.work_dir, 'config.py'))
    commit_id = os.popen('git rev-parse HEAD').readline()
    logger.info(f'Commit ID: {commit_id}')
    runner = SefaRunner(config, logger)
    runner.run()


if __name__ == '__main__':
    main()
