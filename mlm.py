# This file assembles three popular metric learnign baselines, matching network, prototype network and relation network.
# This file is coded based on train_matchingNet.py.
# coding=utf-8
import os
import torch
import torch.optim as optim
import argparse
import time
from transformers import AutoTokenizer

from utils.models import IntentBERT
from utils.IntentDataset import IntentDataset
from utils.Trainer import MLMOnlyTrainer
from utils.Evaluator import FewShotEvaluator
from utils.commonVar import *
from utils.printHelper import *
from utils.tools import *
from utils.Logger import logger
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def parseArgument():
    # ==== parse argument ====
    parser = argparse.ArgumentParser(description='Train with MLM loss')

    # ==== model ====
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--mode', default='multi-class',
                        help='Choose from multi-class')
    parser.add_argument('--tokenizer', default='bert-base-uncased',
                        help="Name of tokenizer")
    parser.add_argument('--LMName', default='bert-base-uncased',
                        help='Name for models and path to saved model')

    # ==== dataset ====
    parser.add_argument('--dataDir',
                        help="Dataset names included in this experiment and separated by comma. "
                        "For example:'OOS,bank77,hwu64'")
    parser.add_argument('--targetDomain',
                        help='Target domain names and separated by comma')

    # ==== evaluation task ====
    parser.add_argument('--way', type=int, default=5)
    parser.add_argument('--shot', type=int, default=2)
    parser.add_argument('--query', type=int, default=5)
    parser.add_argument('--clsFierName', default='Linear',
                        help="Classifer name for few-shot evaluation"
                        "Choose from Linear|SVM|NN|Cosine")

    # ==== optimizer ====
    parser.add_argument('--optimizer', default='Adam',
                        help='Choose from SGD|Adam')
    parser.add_argument('--learningRate', type=float, default=2e-5)
    parser.add_argument('--weightDecay', type=float, default=0)

    # ==== training arguments ====
    parser.add_argument('--disableCuda', action="store_true")
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--taskNum', type=int, default=500)
    
    # ==== other things ====
    parser.add_argument('--loggingLevel', default='INFO',
                        help="python logging level")
    parser.add_argument('--saveModel', action='store_true',
                        help="Whether to save pretrained model")
    parser.add_argument('--saveName', default='none',
                        help="Specify a unique name to save your model"
                        "If none, then there will be a specific name controlled by how the model is trained")
    parser.add_argument('--tensorboard', action='store_true',
                        help="Enable tensorboard to log training and validation accuracy")

    args = parser.parse_args()

    return args

def main():
    # ======= process arguments ======
    args = parseArgument()
    print(args)

    if not args.saveModel:
        logger.info("The model will not be saved after training!")

    # ==== setup logger ====
    if args.loggingLevel == LOGGING_LEVEL_INFO:
        loggingLevel = logging.INFO
    elif args.loggingLevel == LOGGING_LEVEL_DEBUG:
        loggingLevel = logging.DEBUG
    else:
        raise NotImplementedError("Not supported logging level %s", args.loggingLevel)
    logger.setLevel(loggingLevel)

    # ======= process data ======
    # tokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    # load raw dataset
    logger.info(f"Loading data from {args.dataDir}")
    dataset = IntentDataset()
    dataset.loadDataset(splitName(args.dataDir))
    dataset.tokenize(tok)
    # spit data into training, validation and testing
    logger.info("----- Testing Data -----")
    testData = dataset.splitDomain(splitName(args.targetDomain))

    # ======= prepare model ======
    # initialize model
    modelConfig = {}
    modelConfig['device'] = torch.device('cuda:0' if not args.disableCuda else 'cpu')
    modelConfig['clsNumber'] = testData.getLabNum()
    modelConfig['LMName'] = args.LMName
    model = IntentBERT(modelConfig)
    logger.info("----- IntentBERT initialized -----")

    # setup validator
    valParam = {"evalTaskNum": args.taskNum, "clsFierName": args.clsFierName, "multi_label": False}
    valTaskParam = {"way":args.way, "shot":args.shot, "query":args.query}
    tester = FewShotEvaluator(valParam, valTaskParam, testData)

    # setup trainer
    optimizer = None
    if args.optimizer == OPTER_ADAM:
        optimizer = optim.Adam(model.parameters(), lr=args.learningRate, weight_decay=args.weightDecay)
    elif args.optimizer == OPTER_SGD:
        optimizer = optim.SGD(model.parameters(), lr=args.learningRate, weight_decay=args.weightDecay)
    else:
        raise NotImplementedError("Not supported optimizer %s"%(args.optimizer))

    trainingParam = {"epoch"      : args.epochs, \
                     "batch"      : args.batch_size, \
                     "tensorboard": args.tensorboard}
    trainer = MLMOnlyTrainer(trainingParam, optimizer, testData, testData, tester)

    # train
    trainer.train(model, tok)

    # evaluate once more to show results
    tester.evaluate(model, tok, args.mode, logLevel='INFO')

    # save model into disk
    if args.saveModel:
        if args.saveName == 'none':
            prefix = "MLMOnly"
            save_path = os.path.join(SAVE_PATH, f'{prefix}_{args.targetDomain}')
        else:
            save_path = os.path.join(SAVE_PATH, args.saveName)
        logger.info("Saving model.pth into folder: %s", save_path)
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        model.save(save_path)

    # print config
    logger.info(args)
    logger.info(time.asctime())

if __name__ == "__main__":
    main()
    exit(0)
