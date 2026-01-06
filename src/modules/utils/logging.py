import sys
from loguru import logger
from .common import get_rank

def setup_root_logger(debug=False, log_file=None, cmd='train'):
    rank = get_rank()
    rank_info = f'[rank: {rank}] ' if rank >= 0 else ''
    
    logger.remove()
    if rank > 0:
        print(f'{rank_info}Disabling logger message')
    else:
        level = 'DEBUG' if debug else 'INFO'
        print(f'{rank_info}Setting logger level: {level}')
        format = "{time:HH:mm:ss} | {level} | {name}:{function}:{line} - {message}"
        logger.add(sys.stdout, level=level, format=format)
        if log_file and cmd == 'train':
            print(f'{rank_info}Adding logger FileHandler: {log_file}')
            logger.add(log_file, level=level, format=format)
