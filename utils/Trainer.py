from utils.IntentDataset import IntentDataset
from utils.Evaluator import EvaluatorBase
from utils.Logger import logger
from utils.commonVar import *
from utils.tools import mask_tokens, makeTrainExamples
import time
import torch
from torch.utils.data import DataLoader
import numpy as np
import copy
from sklearn.metrics import accuracy_score, r2_score
from torch.utils.tensorboard import SummaryWriter

##
# @brief  base class of trainer
class TrainerBase():
    def __init__(self):
        self.finished=False
        self.bestModelStateDict = None
        self.roundN = 4
        pass

    def round(self, floatNum):
        return round(floatNum, self.roundN)

    def train(self):
        raise NotImplementedError("train() is not implemented.")

    def getBestModelStateDict(self):
        return self.bestModelStateDict

##
# @brief TransferTrainer used to do transfer-training. The training is performed in a supervised manner. All available data is used fo training. By contrast, meta-training is performed by tasks. 
class TransferTrainer(TrainerBase):
    def __init__(self,
                 trainingParam:dict,
                 optimizer,
                 dataset:IntentDataset,
                 unlabeled:IntentDataset,
                 valEvaluator: EvaluatorBase,
                 testEvaluator:EvaluatorBase):
        super(TransferTrainer, self).__init__()
        self.epoch       = trainingParam['epoch']
        self.batch_size  = trainingParam['batch']
        self.validation  = trainingParam['validation']
        self.patience    = trainingParam['patience']
        self.tensorboard = trainingParam['tensorboard']
        self.mlm         = trainingParam['mlm']
        self.lambda_mlm  = trainingParam['lambda mlm']
        self.regression  = trainingParam['regression']

        self.dataset       = dataset
        self.unlabeled     = unlabeled
        self.optimizer     = optimizer
        self.valEvaluator  = valEvaluator
        self.testEvaluator = testEvaluator

        if self.tensorboard:
            self.writer = SummaryWriter()

    def train(self, model, tokenizer, mode='multi-class'):
        self.bestModelStateDict = copy.deepcopy(model.state_dict())
        durationOverallTrain = 0.0
        durationOverallVal = 0.0
        valBestAcc = -1
        accumulateStep = 0

        # evaluate before training
        valAcc, valPre, valRec, valFsc = self.valEvaluator.evaluate(model, tokenizer, mode)
        teAcc, tePre, teRec, teFsc = self.testEvaluator.evaluate(model, tokenizer, mode)
        logger.info('---- Before training ----')
        logger.info("ValAcc %f, Val pre %f, Val rec %f , Val Fsc %f", valAcc, valPre, valRec, valFsc)
        logger.info("TestAcc %f, Test pre %f, Test rec %f, Test Fsc %f", teAcc, tePre, teRec, teFsc)

        if mode == 'multi-class':
            labTensorData = makeTrainExamples(self.dataset.getTokList(), tokenizer, self.dataset.getLabID(), mode=mode)
        else:
            logger.error("Invalid model %d"%(mode))
        dataloader = DataLoader(labTensorData, batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True)
        
        if self.mlm:
            unlabTensorData = makeTrainExamples(self.unlabeled.getTokList(), tokenizer, mode='unlabel')
            unlabeledloader = DataLoader(unlabTensorData, batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True)
            unlabelediter = iter(unlabeledloader)

        for epoch in range(self.epoch):  # an epoch means all sampled tasks are done
            model.train()
            batchTrAccSum     = 0.0
            batchTrLossSPSum  = 0.0
            batchTrLossMLMSum = 0.0
            timeEpochStart    = time.time()

            for batch in dataloader:
                # task data
                Y, ids, types, masks = batch
                X = {'input_ids':ids.to(model.device),
                     'token_type_ids':types.to(model.device),
                     'attention_mask':masks.to(model.device)}

                # forward
                logits = model(X)
                # loss
                if self.regression:
                    lossSP = model.loss_mse(logits, Y.to(model.device))
                else:
                    lossSP = model.loss_ce(logits, Y.to(model.device))

                if self.mlm:
                    try:
                        ids, types, masks = unlabelediter.next()
                    except StopIteration:
                        unlabelediter = iter(unlabeledloader)
                        ids, types, masks = unlabelediter.next()
                    X_un = {'input_ids':ids.to(model.device),
                            'token_type_ids':types.to(model.device),
                            'attention_mask':masks.to(model.device)}
                    mask_ids, mask_lb = mask_tokens(X_un['input_ids'].cpu(), tokenizer)
                    X_un = {'input_ids':mask_ids.to(model.device),
                            'token_type_ids':X_un['token_type_ids'],
                            'attention_mask':X_un['attention_mask']}
                    lossMLM = model.mlmForward(X_un, mask_lb.to(model.device))
                    lossTOT = lossSP + self.lambda_mlm * lossMLM
                else:
                    lossTOT = lossSP

                # backward
                self.optimizer.zero_grad()
                lossTOT.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                self.optimizer.step()

                # calculate train acc
                YTensor = Y.cpu()
                logits = logits.detach().clone()
                if torch.cuda.is_available():
                    logits = logits.cpu()
                if self.regression:
                    predictResult = torch.sigmoid(logits).numpy()
                    acc = r2_score(YTensor, predictResult)
                else:
                    logits = logits.numpy()
                    predictResult = np.argmax(logits, 1)
                    acc = accuracy_score(YTensor, predictResult)

                # accumulate statistics
                batchTrAccSum += acc
                batchTrLossSPSum += lossSP.item()
                if self.mlm:
                    batchTrLossMLMSum += lossMLM.item()

            # current epoch training done, collect data
            durationTrain         = self.round(time.time() - timeEpochStart)
            durationOverallTrain += durationTrain
            batchTrAccAvrg        = self.round(batchTrAccSum/len(dataloader))
            batchTrLossSPAvrg     = batchTrLossSPSum/len(dataloader)
            batchTrLossMLMAvrg    = batchTrLossMLMSum/len(dataloader)
            
            valAcc, valPre, valRec, valFsc = self.valEvaluator.evaluate(model, tokenizer, mode)
            teAcc, tePre, teRec, teFsc     = self.testEvaluator.evaluate(model, tokenizer, mode)

            # display current epoch's info
            logger.info("---- epoch: %d/%d, train_time %f ----", epoch, self.epoch, durationTrain)
            logger.info("SPLoss %f, MLMLoss %f, TrainAcc %f", batchTrLossSPAvrg, batchTrLossMLMAvrg, batchTrAccAvrg)
            logger.info("ValAcc %f, Val pre %f, Val rec %f , Val Fsc %f", valAcc, valPre, valRec, valFsc)
            logger.info("TestAcc %f, Test pre %f, Test rec %f, Test Fsc %f", teAcc, tePre, teRec, teFsc)
            if self.tensorboard:
                self.writer.add_scalar('train loss', batchTrLossSPAvrg+self.lambda_mlm*batchTrLossMLMAvrg, global_step=epoch)
                self.writer.add_scalar('val acc', valAcc, global_step=epoch)
                self.writer.add_scalar('test acc', teAcc, global_step=epoch)

            # early stop
            if not self.validation:
                valAcc = -1
            if (valAcc >= valBestAcc):   # better validation result
                print("[INFO] Find a better model. Val acc: %f -> %f"%(valBestAcc, valAcc))
                valBestAcc = valAcc
                accumulateStep = 0

                # cache current model, used for evaluation later
                self.bestModelStateDict = copy.deepcopy(model.state_dict())
            else:
               accumulateStep += 1
               if accumulateStep > self.patience/2:
                   print('[INFO] accumulateStep: ', accumulateStep)
                   if accumulateStep == self.patience:  # early stop
                       logger.info('Early stop.')
                       logger.debug("Overall training time %f", durationOverallTrain)
                       logger.debug("Overall validation time %f", durationOverallVal)
                       logger.debug("best_val_acc: %f", valBestAcc)
                       break
        
        logger.info("best_val_acc: %f", valBestAcc)


##
# @brief TransferTrainer used to do transfer-training. The training is performed in a supervised manner. All available data is used fo training. By contrast, meta-training is performed by tasks. 
class MLMOnlyTrainer(TrainerBase):
    def __init__(self,
                 trainingParam:dict,
                 optimizer,
                 dataset:IntentDataset,
                 unlabeled:IntentDataset,
                 testEvaluator:EvaluatorBase):
        super(MLMOnlyTrainer, self).__init__()
        self.epoch       = trainingParam['epoch']
        self.batch_size  = trainingParam['batch']
        self.tensorboard = trainingParam['tensorboard']

        self.dataset       = dataset
        self.unlabeled     = unlabeled
        self.optimizer     = optimizer
        self.testEvaluator = testEvaluator

        if self.tensorboard:
            self.writer = SummaryWriter()

    def train(self, model, tokenizer):
        durationOverallTrain = 0.0

        # evaluate before training
        teAcc, tePre, teRec, teFsc = self.testEvaluator.evaluate(model, tokenizer, 'multi-class')
        logger.info('---- Before training ----')
        logger.info("TestAcc %f, Test pre %f, Test rec %f, Test Fsc %f", teAcc, tePre, teRec, teFsc)

        labTensorData = makeTrainExamples(self.dataset.getTokList(), tokenizer, mode='unlabel')
        dataloader = DataLoader(labTensorData, batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True)
        
        for epoch in range(self.epoch):  # an epoch means all sampled tasks are done
            model.train()
            batchTrLossSum = 0.0
            timeEpochStart = time.time()

            for batch in dataloader:
                # task data
                ids, types, masks = batch
                X = {'input_ids':ids.to(model.device),
                     'token_type_ids':types.to(model.device),
                     'attention_mask':masks.to(model.device)}

                # forward
                mask_ids, mask_lb = mask_tokens(X['input_ids'].cpu(), tokenizer)
                X = {'input_ids':mask_ids.to(model.device),
                     'token_type_ids':X['token_type_ids'],
                     'attention_mask':X['attention_mask']}
                loss = model.mlmForward(X, mask_lb.to(model.device))

                # backward
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                self.optimizer.step()

            durationTrain = self.round(time.time() - timeEpochStart)
            durationOverallTrain += durationTrain
            batchTrLossAvrg = batchTrLossSum/len(dataloader)
            
            teAcc, tePre, teRec, teFsc = self.testEvaluator.evaluate(model, tokenizer, 'multi-class')

            # display current epoch's info
            logger.info("---- epoch: %d/%d, train_time %f ----", epoch, self.epoch, durationTrain)
            logger.info("TrainLoss %f", batchTrLossAvrg)
            logger.info("TestAcc %f, Test pre %f, Test rec %f, Test Fsc %f", teAcc, tePre, teRec, teFsc)
            if self.tensorboard:
                self.writer.add_scalar('train loss', batchTrLossAvrg, global_step=epoch)
                self.writer.add_scalar('test acc', teAcc, global_step=epoch)
