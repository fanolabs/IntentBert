from utils.IntentDataset import IntentDataset
from utils.TaskSampler import MultiLabTaskSampler, UniformTaskSampler
from utils.tools import makeEvalExamples
from utils.printHelper import *
from utils.Logger import logger
from utils.commonVar import *
import logging
import torch
import numpy as np
from tqdm import tqdm

from sklearn.metrics import accuracy_score, precision_recall_fscore_support

##
# @brief  base class of evaluator
class EvaluatorBase():
    def __init__(self):
        self.roundN = 4
        pass

    def round(self, floatNum):
        return round(floatNum, self.roundN)

    def evaluate(self):
        raise NotImplementedError("train() is not implemented.")

##
# @brief MetaEvaluator used to do meta evaluation. Tasks are sampled and the model is evaluated task by task.
class FewShotEvaluator(EvaluatorBase):
    def __init__(self, evalParam, taskParam, dataset: IntentDataset):
        super(FewShotEvaluator, self).__init__()
        self.way   = taskParam['way']
        self.shot  = taskParam['shot']
        self.query = taskParam['query']

        self.dataset = dataset

        self.multi_label = evalParam['multi_label']
        self.clsFierName = evalParam['clsFierName']
        self.evalTaskNum = evalParam['evalTaskNum']
        logger.info("In evaluator classifier %s is used.", self.clsFierName)

        if self.multi_label:
            self.taskSampler = MultiLabTaskSampler(self.dataset, self.shot, self.query)
        else:
            self.taskSampler = UniformTaskSampler(self.dataset, self.way, self.shot, self.query)

    def evaluate(self, model, tokenizer, mode='multi-class', logLevel='DEBUG'):
        model.eval()

        performList = []   # acc, pre, rec, fsc
        with torch.no_grad():
            for task in range(self.evalTaskNum):
                # sample a task
                task = self.taskSampler.sampleOneTask()

                # collect data
                supportX = task[META_TASK_SHOT_TOKEN]
                queryX = task[META_TASK_QUERY_TOKEN]
                if mode == 'multi-class':
                    supportY = task[META_TASK_SHOT_LOC_LABID]
                    queryY = task[META_TASK_QUERY_LOC_LABID]
                else:
                    logger.error("Invalid model %d"%(mode))

                # padding
                supportX, supportY, queryX, queryY =\
                    makeEvalExamples(supportX, supportY, queryX, queryY, tokenizer, mode=mode)

                # forward
                queryPrediction = model.fewShotPredict(supportX.to(model.device),
                                                       supportY,
                                                       queryX.to(model.device),
                                                       self.clsFierName,
                                                       mode=mode)
                
                # calculate acc
                acc = accuracy_score(queryY, queryPrediction)   # acc
                if self.multi_label:
                    performDetail = precision_recall_fscore_support(queryY, queryPrediction, average='micro', warn_for=tuple())
                else:
                    performDetail = precision_recall_fscore_support(queryY, queryPrediction, average='macro', warn_for=tuple())

                performList.append([acc, performDetail[0], performDetail[1], performDetail[2]])
        
        # performance mean and std
        performMean = np.mean(np.stack(performList, 0), 0)
        performStd  = np.std(np.stack(performList, 0), 0)

        if logLevel == 'DEBUG':
            itemList = ["acc", "pre", "rec", "fsc"]
            logger.debug("Evaluate statistics: ")
            printMeanStd(performMean, performStd, itemList, debugLevel=logging.DEBUG)
        else:
            itemList = ["acc", "pre", "rec", "fsc"]
            logger.info("Evaluate statistics: ")
            printMeanStd(performMean, performStd, itemList, debugLevel=logging.INFO)

        # acc, pre, rec, F1
        return performMean[0], performMean[1], performMean[2], performMean[3]


##
# @brief MetaEvaluator used to do meta evaluation. Tasks are sampled and the model is evaluated task by task.
class FineTuneEvaluator(EvaluatorBase):
    def __init__(self, evalParam, taskParam, optimizer, dataset: IntentDataset):
        super(FineTuneEvaluator, self).__init__()
        self.way   = taskParam['way']
        self.shot  = taskParam['shot']
        self.query = taskParam['query']

        self.dataset   = dataset
        self.optimizer = optimizer

        self.finetuneSteps = evalParam['finetuneSteps']
        self.evalTaskNum   = evalParam['evalTaskNum']

        self.taskSampler = UniformTaskSampler(self.dataset, self.way, self.shot, self.query)

    def evaluate(self, model, tokenizer, mode='multi-class', logLevel='DEBUG'):
        performList = []   # acc, pre, rec, fsc
        initial_model = model.state_dict().copy()
        initial_optim = self.optimizer.state_dict().copy()

        for task in tqdm(range(self.evalTaskNum)):
            # sample a task
            task = self.taskSampler.sampleOneTask()

            # collect data
            supportX = task[META_TASK_SHOT_TOKEN]
            queryX = task[META_TASK_QUERY_TOKEN]
            if mode == 'multi-class':
                supportY = task[META_TASK_SHOT_LOC_LABID]
                queryY = task[META_TASK_QUERY_LOC_LABID]
            else:
                logger.error("Invalid model %d"%(mode))

            # padding
            supportX, supportY, queryX, queryY =\
                makeEvalExamples(supportX, supportY, queryX, queryY, tokenizer, mode=mode)

            # finetune
            model.train()
            for _ in range(self.finetuneSteps):
                logits = model(supportX.to(model.device))
                loss = model.loss_ce(logits, torch.tensor(supportY).to(model.device))
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                self.optimizer.step()

            model.eval()
            with torch.no_grad():
                if mode == 'multi-class':
                    queryPrediction = model(queryX.to(model.device)).argmax(-1)
                else:
                    logger.error("Invalid model %d"%(mode))
                
                queryPrediction = queryPrediction.cpu().numpy()

                # calculate acc
                acc = accuracy_score(queryY, queryPrediction)   # acc
                performDetail = precision_recall_fscore_support(queryY, queryPrediction, average='macro', warn_for=tuple())

                performList.append([acc, performDetail[0], performDetail[1], performDetail[2]])
            
            model.load_state_dict(initial_model)
            self.optimizer.load_state_dict(initial_optim)
        
        # performance mean and std
        performMean = np.mean(np.stack(performList, 0), 0)
        performStd  = np.std(np.stack(performList, 0), 0)

        if logLevel == 'DEBUG':
            itemList = ["acc", "pre", "rec", "fsc"]
            logger.debug("Evaluate statistics: ")
            printMeanStd(performMean, performStd, itemList, debugLevel=logging.DEBUG)
        else:
            itemList = ["acc", "pre", "rec", "fsc"]
            logger.info("Evaluate statistics: ")
            printMeanStd(performMean, performStd, itemList, debugLevel=logging.INFO)

        # acc, pre, rec, F1
        return performMean[0], performMean[1], performMean[2], performMean[3]
