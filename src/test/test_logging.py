import os
from omegaconf import OmegaConf
from modules.utils.logging import setup_root_logger, logger

def test_logger(conf):
    setup_root_logger(conf.get('debug', False), 
                      conf.get('log_file', None))
    logger.debug('debug message')
    logger.info('info message')
    logger.warning('warning message')
    logger.error('error message')
    
    env_vars = os.environ    
    for key, value in env_vars.items():
        if key.startswith('SLURM'):
            logger.debug(f"{key}: {value}")


if __name__ == '__main__':
    conf = OmegaConf.create({'cmd': 'test_logger'})
    conf.merge_with_cli()
    eval(conf.cmd)(conf)
