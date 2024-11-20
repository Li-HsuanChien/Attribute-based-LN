import random
import argparse, yaml
import cfgs.config as config
from easydict import EasyDict as edict

def parse_args():
    parser = argparse.ArgumentParser(description='HiBERT Args')

    parser.add_argument('--run', dest='run_mode',
                        choices=['train', 'val', 'test', 'gen', 'tsne'],
                        help='{train, val, test,}',
                        type=str, required=True)

    parser.add_argument('--dataset', dest='dataset',
                        # choices=['twt'],
                        help='{tweet}',
                        default='twt', type=str)

    parser.add_argument('--gpu', dest='gpu',
                        help="gpu select, eg.'0, 1, 2'",
                        type=str,
                        default="0, 1")

    parser.add_argument('--no_cuda',
                        action='store_true',
                        default=False,
                        )

    parser.add_argument('--seed', dest='seed',
                        help='fix random seed',
                        type=int,
                        default=random.randint(0, 99999999))

    parser.add_argument('--version', dest='version',
                        help='version control',
                        type=str,
                        default="default")

    args = parser.parse_args()
    return args

# run example:
# python run_PSC.py --run train --dataset imdb --gpu 0,1 --version test
if __name__ == '__main__':
    __C = config.__C

    args = parse_args()
    cfg_file = "cfgs/hietransformers.yml"
    with open(cfg_file, 'r') as f:
        yaml_dict = yaml.load(f, Loader=yaml.FullLoader)

    args_dict = edict({**yaml_dict, **vars(args)})
    config.add_edit(args_dict, __C)
    config.proc(__C)

    print('Hyper Parameters:')
    config.config_print(__C)

    # from trainer.PSC_trainer import Trainer
    # from trainer.PSC_trainer_new import Trainer
    from trainer.PSC_trainer_trunc import Trainer

    execution = Trainer(__C)
    execution.run(__C.run_mode)
