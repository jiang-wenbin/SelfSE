from omegaconf import OmegaConf


def args_parser():
    conf = OmegaConf.create({
        'conf': 'conf/config.yaml',
        'cmd': 'train'
    })
    conf.merge_with_cli()
    conf = OmegaConf.merge(OmegaConf.load(conf.conf), conf)
    OmegaConf.resolve(conf)
    return conf
