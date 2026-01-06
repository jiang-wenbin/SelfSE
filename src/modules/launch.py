from rich import print as rprint
from .utils.args_parser import args_parser
from .utils.init_conf import init_conf
from . import command


def main():
    conf = args_parser()
    conf = init_conf(conf)
    if conf.get('print_conf', False):
        rprint(conf)
    getattr(command, conf['cmd'])(conf)


if __name__ == '__main__':
    main()
