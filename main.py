import torch
import argparse
from experiments.ltf_exp import run_ltf


def get_parser():
    parser = argparse.ArgumentParser(description="wtNet for time series")
    subparser = parser.add_subparsers()
    ltf_subparser = subparser.add_parser("ltf", help="Long-Term_Forecasting")
    ltf_subparser.set_defaults(func=run_ltf)
    ltf_subparser.add_argument("--gpus", type=int, nargs="*", default=0)
    ltf_subparser.add_argument(
        "--dataset",
        type=str,
        nargs="*",
        default="all",
        help="all/etth1/etth2/ettm1/ettm2"
    )
    ltf_subparser.add_argument("--pred_len",
                               type=str,
                               nargs="*",
                               default="all",
                               help="all/96/192/336/720")
    return parser

if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('medium')
    args = get_parser().parse_args()
    # print(args)
    args.func(args)
