#coding=utf-8
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForMaskedLM
from utils.commonVar import *

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

class IntentBERT(nn.Module):
    def __init__(self, config):
        super(IntentBERT, self).__init__()
        self.device = config['device']
        self.LMName = config['LMName']
        self.clsNum = config['clsNumber']
        try:
            self.word_embedding = AutoModelForMaskedLM.from_pretrained(self.LMName)
        except:
            self.word_embedding = AutoModelForMaskedLM.from_pretrained(os.path.join(SAVE_PATH, self.LMName))
        self.linearClsfier = nn.Linear(768, self.clsNum)
        self.dropout = nn.Dropout(0.1) # follow the default in bert model
        # self.word_embedding = nn.DataParallel(self.word_embedding)
        self.word_embedding.to(self.device)
        self.linearClsfier.to(self.device)

    def loss_ce(self, logits, Y):
        loss = nn.CrossEntropyLoss()
        output = loss(logits, Y)
        return output
    
    def loss_mse(self, logits, Y):
        loss = nn.MSELoss()
        output = loss(torch.sigmoid(logits).squeeze(), Y)
        return output

    def loss_kl(self, logits, label):
        # KL-div loss
        probs = F.log_softmax(logits, dim=1)
        # label_probs = F.log_softmax(label, dim=1)
        loss = F.kl_div(probs, label, reduction='batchmean')
        return loss
    
    def forward(self, X):
        # BERT forward
        outputs = self.word_embedding(**X, output_hidden_states=True)

        # extract [CLS] for utterance representation
        CLSEmbedding = outputs.hidden_states[-1][:,0]

        # linear classifier
        CLSEmbedding = self.dropout(CLSEmbedding)
        logits = self.linearClsfier(CLSEmbedding)

        return logits
    
    def mlmForward(self, X, Y):
        # BERT forward
        outputs = self.word_embedding(**X, labels=Y)

        return outputs.loss

    def fewShotPredict(self, supportX, supportY, queryX, clsFierName, mode='multi-class'):
        # calculate word embedding
        # BERT forward
        s_embedding = self.word_embedding(**supportX, output_hidden_states=True).hidden_states[-1]
        q_embedding = self.word_embedding(**queryX, output_hidden_states=True).hidden_states[-1]
        
        # extract [CLS] for utterance representation
        supportEmbedding = s_embedding[:,0]
        queryEmbedding = q_embedding[:,0]
        support_features = self.normalize(supportEmbedding).cpu()
        query_features = self.normalize(queryEmbedding).cpu()

        # select clsfier
        clf = None
        if clsFierName == CLSFIER_LINEAR_REGRESSION:
            clf = LogisticRegression(penalty='l2',
                                     random_state=0,
                                     C=1.0,
                                     solver='lbfgs',
                                     max_iter=1000,
                                     multi_class='multinomial')
            # fit and predict
            clf.fit(support_features, supportY)
        elif clsFierName == CLSFIER_SVM:
            clf = make_pipeline(StandardScaler(), 
                                SVC(gamma='auto',C=1,
                                kernel='linear',
                                decision_function_shape='ovr'))
            # fit and predict
            clf.fit(support_features, supportY)
        elif clsFierName == CLSFIER_MULTI_LABEL:
            clf = MultiOutputClassifier(LogisticRegression(penalty='l2',
                                                           random_state=0,
                                                           C=1.0,
                                                           solver='liblinear',
                                                           max_iter=1000,
                                                           multi_class='ovr',
                                                           class_weight='balanced'))

            clf.fit(support_features, supportY)
        else:
            raise NotImplementedError("Not supported clasfier name %s", clsFierName)
        
        if mode == 'multi-class':
            query_pred = clf.predict(query_features)
        else:
            logger.error("Invalid model %d"%(mode))

        return query_pred
    
    def reinit_clsfier(self):
        self.linearClsfier.weight.data.normal_(mean=0.0, std=0.02)
        self.linearClsfier.bias.data.zero_()
    
    def set_dropout_layer(self, dropout_rate):
        self.dropout = nn.Dropout(dropout_rate)
    
    def set_linear_layer(self, clsNum):
        self.linearClsfier = nn.Linear(768, clsNum)
    
    def normalize(self, x):
        norm = x.pow(2).sum(1, keepdim=True).pow(1. / 2)
        out = x.div(norm)
        return out

    def NN(self, support, support_ys, query):
        """nearest classifier"""
        support = np.expand_dims(support.transpose(), 0)
        query = np.expand_dims(query, 2)

        diff = np.multiply(query - support, query - support)
        distance = diff.sum(1)
        min_idx = np.argmin(distance, axis=1)
        pred = [support_ys[idx] for idx in min_idx]
        return pred

    def CosineClsfier(self, support, support_ys, query):
        """Cosine classifier"""
        support_norm = np.linalg.norm(support, axis=1, keepdims=True)
        support = support / support_norm
        query_norm = np.linalg.norm(query, axis=1, keepdims=True)
        query = query / query_norm

        cosine_distance = query @ support.transpose()
        max_idx = np.argmax(cosine_distance, axis=1)
        pred = [support_ys[idx] for idx in max_idx]
        return pred

    def save(self, path):
        self.word_embedding.save_pretrained(path)
