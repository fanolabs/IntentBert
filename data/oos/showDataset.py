# this script extracts a sub word embedding from the entire one
from gensim.models.keyedvectors import KeyedVectors
import numpy as np
import scipy.io as sio
import nltk
import pdb
import random
import csv
import string
import contractions
import json
import random

def getDomainIntent(domainLabFile):
    domain2lab = {}
    lab2domain = {}
    currentDomain = None
    with open(domainLabFile,'r') as f:
        for line in f:
            if ':' in line and currentDomain == None:
                currentDomain = cleanUpSentence(line)
                domain2lab[currentDomain] = []
            elif line == "\n":
                currentDomain = None
            else:
                intent = cleanUpSentence(line)
                domain2lab[currentDomain].append(intent)

    for key in domain2lab:
        domain = key
        labList = domain2lab[key]
        for lab in labList:
            lab2domain[lab] = domain

    return domain2lab, lab2domain

def cleanUpSentence(sentence):
    # sentence: a string, like " Hello, do you like apple? I hate it!!  "

    # strip
    sentence = sentence.strip()

    # lower case
    sentence = sentence.lower()

    # fix contractions
    sentence = contractions.fix(sentence)

    # remove '_' and '-'
    sentence = sentence.replace('-',' ')
    sentence = sentence.replace('_',' ')

    # remove all punctuations
    sentence = ''.join(ch for ch in sentence if ch not in string.punctuation)

    return sentence
    

def check_data_format(file_path):
    for line in open(file_path,'rb'):
        arr =str(line.strip(),'utf-8')
        arr = arr.split('\t')
        label = [w for w in arr[0].split(' ')]
        question = [w for w in arr[1].split(' ')]

        if len(label) == 0 or len(question) == 0:
            print("[ERROR] Find empty data: ", label, question)
            return False

    return True


def save_data(data, file_path):
    # save data to disk
    with open(file_path, 'w') as f:
        json.dump(data, f)
    return 

def save_domain_intent(data, file_path):
    domain2intent = {}
    for line in data:
        domain = line[0]
        intent = line[1]

        if not domain in domain2intent:
            domain2intent[domain] = set()

        domain2intent[domain].add(intent)

    # save data to disk
    print("Saving domain intent out ... format: domain \t intent")
    with open(file_path,"w") as f:
        for domain in domain2intent:
            intentSet = domain2intent[domain]
            for intent in intentSet:
                f.write("%s\t%s\n" % (domain, intent))
    return 

def display_data(data):
    # dataset count
    print("[INFO] We have %d dataset."%(len(data)))

    datasetName = 'CLINC150'
    data = data[datasetName]

    # domain count
    domainName = set()
    for domain in data:
        domainName.add(domain)
    print("[INFO] There are %d domains."%(len(domainName)))
    print(domainName)

    # intent count
    intentName = set()
    for domain in data:
        for d in data[domain]:
            lab = d[1][0]
            intentName.add(lab)
    intentName = list(intentName)
    intentName.sort()
    print("[INFO] There are %d intent."%(len(intentName)))
    print(intentName)

    # data count
    count = 0
    for domain in data:
        for d in data[domain]:
            count = count+1
    print("[INFO] Data count: %d"%(count))

    # intent for each domain
    domain2intentDict = {}
    for domain in data:
        if not domain in domain2intentDict:
            domain2intentDict[domain] = set()
        
        for d in data[domain]:
            lab = d[1][0]
            domain2intentDict[domain].add(lab)
    print("[INFO] Intent for each domain.")
    print(domain2intentDict)

    # data for each intent
    intent2count = {}
    for domain in data:
        for d in data[domain]:
            lab = d[1][0]
            if not lab in intent2count:
                intent2count[lab] = 0
            intent2count[lab] = intent2count[lab]+1
    print("[INFO] Intent count")
    print(intent2count)

    # examples of data
    exampleNum = 3
    while not exampleNum == 0:
        for domain in data:
            for d in data[domain]:
                lab = d[1]
                utt = d[0]
                if random.random() < 0.001:
                    print("[INFO] Example:--%s, %s, %s, %s"%(datasetName, domain, lab, utt))
                    exampleNum = exampleNum-1
                    break
            if (exampleNum==0):
                break

    return None


##
# @brief clean up data, including intent and utterance
#
# @param data a list of data
#
# @return 
def cleanData(data):
    newData = []
    for d in data:
        utt = d[0]
        lab = d[1]
        
        uttClr = cleanUpSentence(utt)
        labClr = cleanUpSentence(lab)
        newData.append([labClr, uttClr])

    return newData

def constructData(data, intent2domain):
    dataset2domain = {}
    datasetName = 'CLINC150'
    dataset2domain[datasetName] = {}
    for d in data:
        lab = d[0]
        utt = d[1]
        domain = intent2domain[lab]
        if not domain in dataset2domain[datasetName]:
            dataset2domain[datasetName][domain] = []
        dataField = [utt, [lab]]
        dataset2domain[datasetName][domain].append(dataField)

    return dataset2domain


def read_data(file_path):
    with open(file_path) as json_file:
        data = json.load(json_file)
        return data

# read in data
#dataPath = "/data1/haode/projects/EMDIntentFewShot/SPIN_refactor/data/refactor_OOS/dataset.json"
dataPath = "./dataset.json"
print("Loading data ...", dataPath)
# read lines, collect data count for different classes
data = read_data(dataPath)

display_data(data)
print("Display.. done")
