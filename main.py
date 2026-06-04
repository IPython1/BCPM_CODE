import argparse
import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from data_set import DataSet
from model import BCPM
from trainer import Trainer
from trainer_2GPU import Trainer as trainer_2GPU

seed = 2021
np.random.seed(seed)
random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
torch.manual_seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
if __name__ == "__main__":
    parser = argparse.ArgumentParser("Set args", add_help=False)
    parser.add_argument("--embedding_size", type=int, default=64, help="")
    parser.add_argument("--reg_weight", type=float, default=0.001, help="")
    parser.add_argument("--log_reg", type=float, default=0.5, help="")
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--node_dropout", type=float, default=0.75)
    parser.add_argument("--message_dropout", type=float, default=0.25)
    parser.add_argument("--omega", type=float, default=1)
    parser.add_argument("--data_name", type=str, default="mooccube", help="")
    parser.add_argument("--behaviors", help="", action="append")
    parser.add_argument("--loss_type", type=str, default="bpr", help="")
    parser.add_argument("--neg_count", type=int, default=4)
    parser.add_argument("--if_load_model", type=bool, default=False, help="")
    parser.add_argument("--gpu_no", type=int, default=1, help="")
    parser.add_argument("--topk", type=list, default=[5, 10], help="")
    parser.add_argument("--metrics", type=list, default=["hit", "ndcg"], help="")
    parser.add_argument("--lr", type=float, default=0.001, help="")
    parser.add_argument("--decay", type=float, default=0.001, help="")
    parser.add_argument("--batch_size", type=int, default=1024, help="")
    parser.add_argument("--test_batch_size", type=int, default=1024, help="")
    parser.add_argument("--min_epoch", type=int, default=5, help="")
    parser.add_argument("--epochs", type=int, default=200, help="")
    parser.add_argument("--model_path", type=str, default="./check_point", help="")
    parser.add_argument("--check_point", type=str, default="", help="")
    parser.add_argument("--model_name", type=str, default="BCPM", help="")
    parser.add_argument("--device", type=str, default="cuda:0", help="")
    parser.add_argument(
        "--neg_sample_size",
        type=int,
        default=99,
        help="number of negative samples per positive in sampled evaluation (default: 99 = 1pos + 99neg)",
    )
    args = parser.parse_args()
    if args.data_name == "mooccube":
        args.data_path = "./data/MOOCCube"
        args.behaviors = ["browse", "favorite", "study"]
    elif args.data_name == "mooccubex":
        args.data_path = "./data/MOOCCubeX"
        args.behaviors = ["browse", "favorite", "study"]
    else:
        raise Exception("data_name must be mooccube or mooccubex")
    TIME = time.strftime("%Y-%m-%d %H_%M_%S", time.localtime())
    args.TIME = TIME
    logfile = "{}_enb_{}_{}".format(args.data_name, args.embedding_size, TIME)
    os.makedirs(args.model_path, exist_ok=True)
    os.makedirs("./log/{}".format(args.model_name), exist_ok=True)
    logger.add("./log/{}/{}.log".format(args.model_name, logfile), encoding="utf-8")
    start = time.time()
    dataset = DataSet(args)
    model = BCPM(args, dataset).to(args.device)
    if args.gpu_no == 2:
        model = nn.DataParallel(model, device_ids=[0, 1])
        trainer = trainer_2GPU(model, dataset, args)
    else:
        trainer = Trainer(model, dataset, args)
    logger.info(args.__str__())
    logger.info(model)
    trainer.train_model()
    logger.info("train end total cost time: {}".format(time.time() - start))
