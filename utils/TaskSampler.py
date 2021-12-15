from utils.IntentDataset import IntentDataset
from utils.Logger import logger
import random
from utils.commonVar import *
import numpy as np
import copy
from sklearn.preprocessing import MultiLabelBinarizer
# random.seed(0)
# base class for task samper
# sample meta-task from a dataset for training and evaluation
class TaskSampler():
    def __init__(self, dataset:IntentDataset):
        self.dataset = dataset

    def sampleOneTask():
        raise NotImplementedError("sampleOneTask() is not implemented.")

class UniformTaskSampler(TaskSampler):
    def __init__(self, dataset:IntentDataset, way, shot, query):
        super(UniformTaskSampler, self).__init__(dataset)
        self.way = way
        self.shot = shot
        self.query = query
        self.taskPool = None
        self.dataset = dataset

    ##
    # @brief sample data index for a task. Class global IDs are also sampled.
    #
    # @return a dict, print it to see what's there
    def sampleClassIDsDataInd(self):
        taskInfo = {}
        glbLabNum = self.dataset.getLabNum()
        labID2DataInd  = self.dataset.getLabID2dataInd()

        uniqueGlbLabIDs = list(range(glbLabNum))
        # random sample global label IDs
        taskGlbLabIDs = random.sample(uniqueGlbLabIDs, self.way)
        taskInfo[META_TASK_GLB_LABID] = taskGlbLabIDs

        # sample data for each labID
        taskInfo[META_TASK_SHOT_GLB_LABID] = []
        taskInfo[META_TASK_QUERY_GLB_LABID] = []
        taskInfo[META_TASK_SHOT_DATAIND] = []
        taskInfo[META_TASK_QUERY_DATAIND] = []
        for labID in taskGlbLabIDs:
            # random sample support data and query data
            dataInds = random.sample(labID2DataInd[labID], self.shot+self.query)
            random.shuffle(dataInds)
            shotDataInds = dataInds[:self.shot]
            queryDataInds = dataInds[(-self.query):]

            taskInfo[META_TASK_SHOT_GLB_LABID].extend([labID]*(self.shot))
            taskInfo[META_TASK_SHOT_DATAIND].extend(shotDataInds)
            taskInfo[META_TASK_QUERY_GLB_LABID].extend([labID]*(self.query))
            taskInfo[META_TASK_QUERY_DATAIND].extend(queryDataInds)

        return taskInfo


    ##
    # @brief it works with sampleClassIDsDataInd(), taking a taskInfo returned by sampleClassIDsDataInd(), then return data in the task. 
    #
    # @param taskDataInds a dict containing data index
    #
    # @return a dict containing task data, print it to see what's there
    def collectDataForTask(self, taskDataInds):
        task = {}

        # compose local labID from glbLabID
        glbLabIDList = taskDataInds[META_TASK_GLB_LABID]
        glbLabID2LocLabID = {}
        for pos, glbLabID in enumerate(glbLabIDList):
            glbLabID2LocLabID[glbLabID] = pos
        tokList = self.dataset.getTokList()
        labList = self.dataset.getLabList()

        # support
        task[META_TASK_SHOT_LOC_LABID] = [glbLabID2LocLabID[glbLabID] for glbLabID in taskDataInds[META_TASK_SHOT_GLB_LABID]]
        task[META_TASK_SHOT_TOKEN]     = [tokList[i] for i in taskDataInds[META_TASK_SHOT_DATAIND]]
        task[META_TASK_SHOT_LAB]       = [labList[i] for i in taskDataInds[META_TASK_SHOT_DATAIND]]

        # query
        task[META_TASK_QUERY_LOC_LABID] = [glbLabID2LocLabID[glbLabID] for glbLabID in taskDataInds[META_TASK_QUERY_GLB_LABID]]
        task[META_TASK_QUERY_TOKEN] = [tokList[i] for i in taskDataInds[META_TASK_QUERY_DATAIND]]
        task[META_TASK_QUERY_LAB] = [labList[i] for i in taskDataInds[META_TASK_QUERY_DATAIND]]

        return task
 
    def sampleOneTask(self):
        # 1. sample classes and data index
        taskDataInds = self.sampleClassIDsDataInd()

        # 2. according to data index, select data, such tokens, lenths, label names, etc.
        task = self.collectDataForTask(taskDataInds)

        return task


class MultiLabTaskSampler(TaskSampler):
    def __init__(self, dataset:IntentDataset, shot, query):
        super(MultiLabTaskSampler, self).__init__(dataset)
        self.shot = shot
        self.query = query
        self.taskPool = None
        self.dataset = dataset
    
    ##
    # @brief sample data index for a task. Class global IDs are also sampled.
    #
    # @return a dict, print it to see what's there
    def sampleClassIDsDataInd(self):
        taskInfo = {}
        glbLabNum = self.dataset.getLabNum()
        labID2DataInd = self.dataset.getLabID2dataInd()

        uniqueGlbLabIDs = list(range(glbLabNum))
        # random sample global label IDs
        taskInfo[META_TASK_GLB_LABID] = uniqueGlbLabIDs

        # sample data for each labID
        taskInfo[META_TASK_SHOT_GLB_LABID] = []
        taskInfo[META_TASK_QUERY_GLB_LABID] = []
        taskInfo[META_TASK_SHOT_DATAIND] = []
        taskInfo[META_TASK_QUERY_DATAIND] = []
        shotDataInds, queryDataInds = [], []
        for labID in uniqueGlbLabIDs:
            # random sample support data
            dataInds = random.sample(labID2DataInd[labID], self.shot)
            shotDataInds += dataInds
            # random sample query data
            remain = list(set(labID2DataInd[labID])-set(dataInds))
            dataInds = random.sample(remain, self.query)
            queryDataInds += dataInds
        shotDataInds = list(set(shotDataInds))
        queryDataInds = list(set(queryDataInds))
        shotDataInds = self.checkForDuplicate(shotDataInds, required_num=self.shot)
        queryDataInds = self.checkForDuplicate(queryDataInds, required_num=self.query)
        
        taskInfo[META_TASK_SHOT_DATAIND] = shotDataInds
        taskInfo[META_TASK_QUERY_DATAIND] = queryDataInds

        dataInd2LabID = self.dataset.getDataInd2labID()
        for d in shotDataInds:
            taskInfo[META_TASK_SHOT_GLB_LABID].append(dataInd2LabID[d])
        for d in queryDataInds:
            taskInfo[META_TASK_QUERY_GLB_LABID].append(dataInd2LabID[d])
        return taskInfo
    
    def checkForDuplicate(self, dataInds, required_num):
        dataInd2LabID = self.dataset.getDataInd2labID()
        label_lists = []
        for di in dataInds:
            label_lists.extend(dataInd2LabID[di])
        label_names, counts = np.unique(label_lists, return_counts=True)
        shot_counts = {ln: c for ln, c in zip(label_names, counts)}
        loopInds = copy.deepcopy(dataInds)
        for di in loopInds:
            can_remove = True
            for l in dataInd2LabID[di]:
                if (l in shot_counts) and (shot_counts[l] - 1 < required_num):
                    can_remove = False
            if can_remove:
                dataInds.remove(di)
                for l in dataInd2LabID[di]:
                    shot_counts[l] -= 1
        return dataInds

    ##
    # @brief it works with sampleClassIDsDataInd(), taking a taskInfo returned by sampleClassIDsDataInd(), then return data in the task. 
    #
    # @param taskDataInds a dict containing data index
    #
    # @return a dict containing task data, print it to see what's there
    def collectDataForTask(self, taskDataInds):
        task = {}

        # compose local labID from glbLabID
        tokList = self.dataset.getTokList()
        labList = self.dataset.getLabList()

        mlb = MultiLabelBinarizer()

        # support
        task[META_TASK_SHOT_LOC_LABID] = mlb.fit_transform(taskDataInds[META_TASK_SHOT_GLB_LABID]).tolist()
        task[META_TASK_SHOT_TOKEN]     = [tokList[i] for i in taskDataInds[META_TASK_SHOT_DATAIND]]
        task[META_TASK_SHOT_LAB]       = [labList[i] for i in taskDataInds[META_TASK_SHOT_DATAIND]]

        # query
        task[META_TASK_QUERY_LOC_LABID] = mlb.fit_transform(taskDataInds[META_TASK_QUERY_GLB_LABID]).tolist()
        task[META_TASK_QUERY_TOKEN] = [tokList[i] for i in taskDataInds[META_TASK_QUERY_DATAIND]]
        task[META_TASK_QUERY_LAB] = [labList[i] for i in taskDataInds[META_TASK_QUERY_DATAIND]]

        return task
 
    def sampleOneTask(self):
        # 1. sample classes and data index
        taskDataInds = self.sampleClassIDsDataInd()

        # 2. according to data index, select data, such tokens, lenths, label names, etc.
        task = self.collectDataForTask(taskDataInds)

        return task