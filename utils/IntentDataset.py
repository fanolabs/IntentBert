#coding=utf-8
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

from utils.commonVar import *
import json
from utils.Logger import logger
import os
import copy
import random


class IntentDataset():
    def __init__(self,
                 domList=None,
                 labList=None,
                 uttList=None,
                 tokList=None,
                 regression=False,
                 multi_label=False):
        self.regression  = regression
        self.multi_label = multi_label

        self.domList = [] if domList is None else domList
        self.labList = [] if labList is None else labList
        self.uttList = [] if uttList is None else uttList
        self.tokList = [] if tokList is None else tokList

        if (self.labList is not None) and (not self.regression):
            self.createLabID()
        if self.regression:
            self.labIDList = self.labList
        if not self.multi_label:
            self.convertLabs()
        self.labID2DataInd = None
        self.dataInd2LabID = None

    def getDomList(self):
        return self.domList

    def getLabList(self):
        return self.labList
    
    def getUttList(self):
        return self.uttList
    
    def getTokList(self):
        return self.tokList

    def getAllData(self):
        return self.domList, self.labList, self.uttList, self.tokList
    
    def getLabNum(self):
        labSet = set()
        for lab in self.labList:
            if self.multi_label:
                for l in lab:
                    labSet.add(l)
            else:
                labSet.add(lab)
        return len(labSet)
    
    def getLabID(self):
        return self.labIDList

    def checkData(self, utt: str, label: str):
        if not self.regression:
            if len(label) == 0 or len(utt) == 0:
                logger.warning("Illegal label %s or utterance %s, 0 length", label, utt)
                return 1
        return 0

    def loadDataset(self, dataDirList):
        self.domList, self.labList, self.uttList = [], [], []
        dataFilePathList = \
            [os.path.join(DATA_PATH, dataDir, FILE_NAME_DATASET) for dataDir in dataDirList]
        
        dataList = []
        for dataFilePath in dataFilePathList:
            with open(dataFilePath, 'r') as json_file:
                dataList.append(json.load(json_file))

        delDataNum = 0
        for data in dataList:
            for datasetName in data:
                dataset = data[datasetName]
                for domainName in dataset:
                    domain = dataset[domainName]
                    for dataItem in domain:
                        utt = dataItem[0]
                        labList = dataItem[1]

                        if self.multi_label:
                            lab = labList
                        else:
                            lab = labList[0]
                        
                        if not self.checkData(utt, lab) == 0:
                            logger.warning("Illegal label %s or utterance %s, too short length", lab, utt)
                            delDataNum = delDataNum+1
                        else:
                            self.domList.append(domainName)
                            self.labList.append(lab)
                            self.uttList.append(utt)

        # report deleted data number 
        if (delDataNum>0):
            logger.warning("%d data is deleted from dataset.", delDataNum)

        # sanity check
        countSet = set()
        countSet.add(len(self.domList))
        countSet.add(len(self.labList))
        countSet.add(len(self.uttList))
        if len(countSet) > 1:
            logger.error("Unaligned data list. Length of data list: dataset %d, domain %d, lab %d, utterance %d", len(self.domainList), len(self.labList), len(self.uttList))
            exit(1)
        if not self.regression:
            self.createLabID()
        else:
            self.labIDList = self.labList
        logger.info(f"{countSet} data collected")
        return 0

    def removeStopWord(self):
        raise NotImplementedError 

        # print info
        logger.info("Removing stop words ...")
        logger.info("Before removing stop words: data count is %d", len(self.uttList))

        # remove stop word
        stopwordsEnglish = stopwords.words('english')
        uttListNew = []
        labListNew = []
        delLabListNew = []
        delUttListNew = []  # Utt for utterance
        maxLen = -1
        for lab, utt in zip(self.labList, self.uttList):
            uttWordListNew = [w for w in utt.split(' ') if not word in stopwordsEnglish]
            uttNew = ' '.join(uttWordListNew)

            uttNewLen = len(uttWordListNew)
            if uttNewLen <= 0:   # too short utterance, delete it from dataset
                delLabListNew.append(lab)
                delUttListNew.append(uttNew)
            else:   # utt with normal length
                if uttNewLen > maxLen:
                    maxLen = uttNewLen
                labListNew.append(lab)
                uttListNew.append(uttNew)
        self.labList = labListNew
        self.uttListNew = uttListNew
        self.delLabList.append(delLabListNew)
        self.delUttList.append(delUttListNew)

        # update data list
        logger.info("After removing stop words: data count is %d", len(self.uttList))
        logger.info("Removing stop words ... done.")
      
        return 0

    def splitDomain(self, domainName: list, regression=False, multi_label=False):
        domList = self.getDomList()

        # collect index
        indList = []
        for ind, domain in enumerate(domList):
            if domain in domainName:
                indList.append(ind)

        # sanity check
        dataCount = len(indList)
        if dataCount<1:
            logger.error("Empty data for domain %s", domainName)
            exit(1)
        
        logger.info("For domain %s, %d data is selected from %d data in the dataset.", domainName, dataCount, len(domList))
        
        # get all data from dataset
        domList, labList, uttList, tokList = self.getAllData()
        domDomList = [domList[i] for i in indList]
        domLabList = [labList[i] for i in indList]
        domUttList = [uttList[i] for i in indList]
        if self.tokList:
            domTokList = [tokList[i] for i in indList]
        else:
            domTokList = []
        domDataset = IntentDataset(domDomList, domLabList, domUttList, domTokList, regression=regression, multi_label=multi_label)

        return domDataset
    
    def tokenize(self, tokenizer):
        self.tokList = []
        for u in self.uttList:
            ut = tokenizer(u)
            if 'token_type_ids' not in ut:
                ut['token_type_ids'] = [0]*len(ut['input_ids'])
            self.tokList.append(ut)
    
    def shuffle_words(self):
        newList = []
        for u in self.uttList:
            replace = copy.deepcopy(u)
            replace = replace.split(' ')
            random.shuffle(replace)
            replace = ' '.join(replace)
            newList.append(replace)
        self.uttList = newList
    
    # convert label names to label IDs: 0, 1, 2, 3
    def createLabID(self):
        # get unique label
        labSet = set()
        for lab in self.labList:
            if self.multi_label:
                for l in lab:
                    labSet.add(l)
            else:
                labSet.add(lab)
        
        # get number
        self.labNum = len(labSet)
        sortedLabList = list(labSet)
        sortedLabList.sort()

        # fill up dict: lab -> labID
        self.name2LabID = {}
        for ind, lab in enumerate(sortedLabList):
            if not lab in self.name2LabID:
                self.name2LabID[lab] = ind

        # fill up label ID list
        self.labIDList =[]
        for lab in self.labList:
            if self.multi_label:
                labID = []
                for l in lab:
                    labID.append(self.name2LabID[l])
                self.labIDList.append(labID)
            else:
                self.labIDList.append(self.name2LabID[lab])

        # sanity check
        if not len(self.labIDList) == len(self.uttList):
            logger.error("create labID error. Not consistence labe ID list length and utterance list length.")
            exit(1)
        
    def getLabID2dataInd(self):
        if not self.labID2DataInd == None:
            return self.labID2DataInd
        else:
            self.labID2DataInd = {}
            for dataInd, labID in enumerate(self.labIDList):
                if self.multi_label:
                    for l in labID:
                        if not l in self.labID2DataInd:
                            self.labID2DataInd[l] = []
                        self.labID2DataInd[l].append(dataInd)
                else:
                    if not labID in self.labID2DataInd:
                        self.labID2DataInd[labID] = []
                    self.labID2DataInd[labID].append(dataInd)
            
            # sanity check
            if not self.multi_label:
                dataCount = 0
                for labID in self.labID2DataInd:
                    dataCount = dataCount + len(self.labID2DataInd[labID])
                if not dataCount == len(self.uttList):
                    logger.error("Inconsistent data count %d and %d when generating dict, labID2DataInd", dataCount, len(self.uttList))
                    exit(1)

            return self.labID2DataInd
    
    def getDataInd2labID(self):
        if not self.dataInd2LabID == None:
            return self.dataInd2LabID
        else:
            self.dataInd2LabID = {}
            for dataInd, labID in enumerate(self.labIDList):
                self.dataInd2LabID[dataInd] = labID
        return self.dataInd2LabID
    
    def convertLabs(self):
        # when the dataset is not multi-label, convert labels from list to a single instance
        if self.labList:
            if isinstance(self.labList[0], list):
                newList = []
                for l in self.labList:
                    newList.append(l[0])
                self.labList = newList
        if self.labIDList:
            if isinstance(self.labIDList[0], list):
                newList = []
                for l in self.labIDList:
                    newList.append(l[0])
                self.labIDList = newList