import os
import pathlib
import re

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_score, recall_score
from sklearn import metrics

AAs = np.array(list('WFGAVILMPYSTNQCKRHDE'))
all_dict = {'W': 0, 'F': 1, 'G': 2, 'A': 3, 'V': 4, 'I': 5, 'L': 6, 'M': 7, 'P': 8, 'Y': 9, 'S': 10, 'T': 11, 'N': 12,
            'Q': 13, 'C': 14, 'K': 15, 'R': 16, 'H': 17, 'D': 18, 'E': 19}

curPath = os.getcwd()
##AAidx_file='AAindexNormalized.txt' ## AA index reached AUC about 61% for L=14. Worse than AdaBoost
##AAidx_file='AtchleyFactors.txt'  ## Atchley factors work worse than using 544 AA index
AAidx_file = 'AAidx_PCA.txt'  ## works like a charm!!!


AAidx_Dict = {}


def GetFeatureLabels(TumorCDR3s, NonTumorCDR3s):
    nt = len(TumorCDR3s)
    nc = len(NonTumorCDR3s)
    LLt = [len(ss) for ss in TumorCDR3s]
    LLt = np.array(LLt)

    LLc = [len(ss) for ss in NonTumorCDR3s]
    LLc = np.array(LLc)
    NL = range(12, 18)
    FeatureDict = {}
    LabelDict = {}
    for LL in NL:
        vvt = np.where(LLt == LL)[0]

        vvc = np.where(LLc == LL)[0]
        Labels = [1] * len(vvt) + [0] * len(vvc)
        Labels = np.array(Labels)
        Labels = Labels.astype(np.int32)
        data = []
        for ss in TumorCDR3s[vvt]:
            if len(pat.findall(ss)) > 0:
                continue
            data.append(AAindexEncoding(ss))
        #            data.append(OneHotEncoding(ss))
        for ss in NonTumorCDR3s[vvc]:
            if len(pat.findall(ss)) > 0:
                continue
            data.append(AAindexEncoding(ss))
        #            data.append(OneHotEncoding(ss))
        data = np.array(data)
        features = {'x': data, 'LL': LL}
        FeatureDict[LL] = features
        LabelDict[LL] = Labels
    return FeatureDict, LabelDict


def caculateAUC(AUC_outs, AUC_labels):
    ROC = 0
    outs = []
    labels = []
    for (index, AUC_out) in enumerate(AUC_outs):
        softmax = nn.Softmax(dim=1)
        out = softmax(AUC_out).detach().numpy()
        out = out[:, 1]
        for out_one in out.tolist():
            outs.append(out_one)
        for AUC_one in AUC_labels[index].tolist():
            labels.append(AUC_one)

    outs = np.array(outs)


    labels = np.array(labels)

    fpr, tpr, thresholds = metrics.roc_curve(labels, outs, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    aupr = metrics.average_precision_score(labels, outs)
    outs= np.where(outs > 0.5, 1, 0)
    precision = precision_score(labels, outs)

    recall = recall_score(labels, outs)


    return auc, aupr,    precision,    recall


def AAindexEncoding(Seq):
    Ns = len(Seq)

    AAE = np.zeros([Ns - 2])

    for kk in range(Ns):
        if kk == 0 or kk == (Ns - 1):
            a = 0
        else:
            ss = Seq[kk]

            AAE[kk - 1] = all_dict[ss]
    for kk in range(15 - Ns + 2):
        AAE = np.append(AAE, values=20)

    AAE = np.transpose(AAE.astype(np.int))

    return AAE






pat = re.compile('[\\*_XB]')  ## non-productive CDR3 patterns


class Mydata(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __getitem__(self, index):
        gene = torch.from_numpy(self.x[index])

        # gene=gene.float()

        return gene, int(self.y[index])

    def __len__(self):
        return self.x.shape[0]

class Advanced_feature_extractor(nn.Module):

    def __init__(self):
        super(Advanced_feature_extractor, self).__init__()
        self.embedding = nn.Embedding(21, 20)

        self.conv1 = nn.Conv1d(30, 1200, kernel_size=(510,), stride=1)
        nn.init.xavier_normal_(self.conv1.weight, gain=1)

        self.conv2 = nn.Conv1d(30, 900, kernel_size=(510,), stride=1)
        nn.init.xavier_normal_(self.conv2.weight, gain=1)

        self.Flatten = nn.Flatten()

        self.relu = nn.ReLU()
        self.linear1 = nn.Linear(2100, 1000)
        nn.init.xavier_normal_(self.linear1.weight, gain=1)

        self.linear2 = nn.Linear(1000, 2)
        nn.init.xavier_normal_(self.linear1.weight, gain=1)

        # self.linear3=nn.Linear(100,2)
        self.BatchNorm1 = nn.LazyBatchNorm1d()

        self.BatchNorm2 = nn.LazyBatchNorm1d()
        self.dropout = nn.Dropout(0.3)  # best=0.3

    def forward(self, input):
        # food_conv1 = self.maxpool1(self.dropout(self.BatchNorm(self.relu(self.food_conv1(input)))).squeeze(3))
        conv1 = self.conv1(input)

        conv1 = self.relu(conv1)
        conv1 = self.BatchNorm1(conv1)
        conv1 = self.dropout(conv1)

        #

        conv2 = self.conv2(input)
        conv2 = self.relu(conv2)
        conv2 = self.BatchNorm2(conv2)
        conv2 = self.dropout(conv2)

        all = torch.cat([conv1, conv2], 1).squeeze(2)

        all = self.linear2(self.relu(self.linear1(all)))

        return all


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=15):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)

        position = torch.arange(max_len).reshape((-1, 1))
        div_term = torch.pow(10000, torch.arange(0, d_model, 2).reshape((1, -1)) / d_model)
        # sin and cos position encoding

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x

class PCTE(nn.Module):
    def __init__(self, hidden_dim=510,
                 dim_feedforward=1024, num_head=10, num_layers=3, dropout=0.3, max_len=15, activation: str = "relu"):
        super(PCTE, self).__init__()

        self.embeddings = nn.Embedding(21, 510)
        self.conv1 = nn.Conv1d(15, 15, kernel_size=(3,), stride=1)
        nn.init.xavier_normal_(self.conv1.weight, gain=1)
        self.conv2 = nn.Conv1d(15, 15, kernel_size=(5,), stride=1)
        nn.init.xavier_normal_(self.conv2.weight, gain=1)
        self.position_embedding = PositionalEncoding(hidden_dim, dropout, max_len)

        encoder_layer_one = nn.TransformerEncoderLayer(hidden_dim, 10, dim_feedforward, dropout, activation=activation)
        self.transformer_one = nn.TransformerEncoder(encoder_layer_one, num_layers)

        encoder_layer_two = nn.TransformerEncoderLayer(hidden_dim, 5, dim_feedforward, dropout, activation=activation)
        self.transformer_two = nn.TransformerEncoder(encoder_layer_two, num_layers)

        self.output = nn.Linear(hidden_dim, 2)
        self.output2 = nn.Linear(hidden_dim, 2)
        self.relu = nn.ReLU()
        self.BatchNorm1 = nn.LazyBatchNorm1d()
        self.BatchNorm2 = nn.LazyBatchNorm1d()
        self.dropout = nn.Dropout(dropout)
        self.afe = Advanced_feature_extractor()

    def forward(self, inputs):
        attention_mask = self.get_key_padding_mask(tokens=inputs)

        hidden_states = self.embeddings(inputs)

        hidden_states = torch.transpose(hidden_states, 0, 1)
        hidden_states = self.position_embedding(hidden_states)

        attention_mask = attention_mask.cuda()
        hidden_states_one = self.transformer_one(hidden_states, src_key_padding_mask=attention_mask)
        hidden_states_two = self.transformer_two(hidden_states, src_key_padding_mask=attention_mask)
        hidden_states_one = torch.transpose(hidden_states_one, 0, 1)
        hidden_states_two = torch.transpose(hidden_states_two, 0, 1)

        tf = torch.cat([hidden_states_one, hidden_states_two], 1)

        # time series feature

        return self.afe(tf)

    def get_key_padding_mask(self, tokens):
        key_padding_mask = torch.ones(tokens.size())
        key_padding_mask[tokens == 20] = 0
        inv_key_padding_mask = (1 - key_padding_mask)

        return inv_key_padding_mask.to(torch.bool)

def Train_Test(ftumor, fnormal, feval_tumor, feval_normal, rate=0.33, n=100, STEPs=10000, dir_prefix=curPath + '/tmp'):
    all_acc = 0.0
    all_auc = 0.0
    ## rate: cross validation ratio: 0.2 means 80% samples will be used for training
    ## n: number of subsamplings
    pathlib.Path(dir_prefix).mkdir(parents=True, exist_ok=True)
    tumorCDR3s = []
    g = open(ftumor)
    for ll in g.readlines():
        rr = ll.strip()
        if not rr.startswith('C') or not rr.endswith('F'):
            print("Non-standard CDR3s. Skipping.")
            continue
        tumorCDR3s.append(rr)
    normalCDR3s = []
    g = open(fnormal)
    for ll in g.readlines():
        rr = ll.strip()
        if not rr.startswith('C') or not rr.endswith('F'):
            print("Non-standard CDR3s. Skipping.")
            continue
        normalCDR3s.append(rr)
    count = 0
    nt = len(tumorCDR3s)

    nn = len(normalCDR3s)

    vt_idx = range(0, nt)
    vn_idx = range(0, nn)
    nt_s = int(np.ceil(nt * (1 - rate)))

    nn_s = int(np.ceil(nn * (1 - rate)))
    PredictClassList = []
    PredictLabelList = []
    AUCDictList = []

    while count < n:
        print("==============Training cycle %d.=============" % (count))
        ID = str(count)
        vt_train = np.random.choice(vt_idx, nt_s, replace=False)
        vt_test = [x for x in vt_idx if x not in vt_train]
        vn_train = np.random.choice(vn_idx, nn_s, replace=False)
        vn_test = [x for x in vn_idx if x not in vn_train]
        sTumorTrain = np.array(tumorCDR3s)[vt_train]

        sNormalTrain = np.array(normalCDR3s)[vn_train]
        sTumorTest = np.array(tumorCDR3s)[vt_test]
        sNormalTest = np.array(normalCDR3s)[vn_test]
        ftrain_tumor = dir_prefix + '/sTumorTrain-' + str(ID) + '.txt'

        ftrain_normal = dir_prefix + '/sNormalTrain-' + str(ID) + '.txt'
        feval_tumor = dir_prefix + '/sTumorTest-' + str(ID) + '.txt'
        feval_normal = dir_prefix + '/sNormalTest-' + str(ID) + '.txt'
        h = open(ftrain_tumor, 'w')
        _ = [h.write(x + '\n') for x in sTumorTrain]
        h.close()
        h = open(ftrain_normal, 'w')
        _ = [h.write(x + '\n') for x in sNormalTrain]
        h.close()
        h = open(feval_tumor, 'w')
        _ = [h.write(x + '\n') for x in sTumorTest]
        h.close()
        h = open(feval_normal, 'w')
        _ = [h.write(x + '\n') for x in sNormalTest]
        h.close()
        g = open(ftrain_tumor)

        Train_Tumor = []
        for line in g.readlines():
            Train_Tumor.append(line.strip())
        Train_Tumor = np.array(Train_Tumor)

        g = open(ftrain_normal)
        Train_Normal = []
        for line in g.readlines():
            Train_Normal.append(line.strip())
        Train_Normal = np.array(Train_Normal)
        TrainFeature, TrainLabels = GetFeatureLabels(Train_Tumor, Train_Normal)

        g = open(feval_tumor)
        Eval_Tumor = []
        for line in g.readlines():
            Eval_Tumor.append(line.strip())
        Eval_Tumor = np.array(Eval_Tumor)
        g = open(feval_normal)
        Eval_Normal = []
        for line in g.readlines():
            Eval_Normal.append(line.strip())
        Eval_Normal = np.array(Eval_Normal)
        EvalFeature, EvalLabels = GetFeatureLabels(Eval_Tumor, Eval_Normal)


        count = count + 1
        Train_data = []
        for x in TrainFeature[12]["x"]:
            Train_data.append(x)
        for x in TrainFeature[13]["x"]:
            Train_data.append(x)
        for x in TrainFeature[14]["x"]:
            Train_data.append(x)
        for x in TrainFeature[15]["x"]:
            Train_data.append(x)
        for x in TrainFeature[16]["x"]:
            Train_data.append(x)
        for x in TrainFeature[17]["x"]:
            Train_data.append(x)
        Train_data = np.array(Train_data)

        Train_label = []
        for y in TrainLabels[12]:
            Train_label.append(y)
        for y in TrainLabels[13]:
            Train_label.append(y)
        for y in TrainLabels[14]:
            Train_label.append(y)
        for y in TrainLabels[15]:
            Train_label.append(y)
        for y in TrainLabels[16]:
            Train_label.append(y)
        for y in TrainLabels[17]:
            Train_label.append(y)
        Train_label = np.array(Train_label)



        Eval_data = []
        for x in EvalFeature[12]["x"]:
            Eval_data.append(x)
        for x in EvalFeature[13]["x"]:
            Eval_data.append(x)
        for x in EvalFeature[14]["x"]:
            Eval_data.append(x)
        for x in EvalFeature[15]["x"]:
            Eval_data.append(x)
        for x in EvalFeature[16]["x"]:
            Eval_data.append(x)
        for x in EvalFeature[17]["x"]:
            Eval_data.append(x)
        Eval_data = np.array(Eval_data)

        Eval_label = []
        for y in EvalLabels[12]:
            Eval_label.append(y)
        for y in EvalLabels[13]:
            Eval_label.append(y)
        for y in EvalLabels[14]:
            Eval_label.append(y)
        for y in EvalLabels[15]:
            Eval_label.append(y)
        for y in EvalLabels[16]:
            Eval_label.append(y)
        for y in EvalLabels[17]:
            Eval_label.append(y)
        Eval_label = np.array(Eval_label)


        mydata_train_12 = Mydata(Train_data, Train_label)
        dataloader_train_12 = DataLoader(dataset=mydata_train_12, batch_size=200, shuffle=True)
        mydata_test_12 = Mydata(Eval_data, Eval_label)
        dataloader_test_12 = DataLoader(dataset=mydata_test_12, batch_size=200, shuffle=True)
        model = PCTE()
        model = model.cuda()

        optimizer = torch.optim.AdamW(params=model.parameters(), lr=0.000001, weight_decay=5e-3)

        train_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[300], gamma=0.1)



        loss = torch.nn.CrossEntropyLoss()
        for i in range(1000):
            # train_scheduler.step()
            test_acc = 0.0
            train_acc = 0.0
            model.train()
            for data in dataloader_train_12:
                input, label = data
                input = input.cuda()
                label = label.cuda()
                out = model(input)
                out = out.cuda()
                optimizer.zero_grad()
                result_loss = loss(out, label)

                result_loss = result_loss.cuda()
                result_loss.backward()
                optimizer.step()
                train_acc = train_acc + ((out.argmax(1) == label).sum())
                # print(i)
            # print("train")
            train_scheduler.step()
            # train_scheduler2.step()

            # train_scheduler.step()
            # print(float(train_acc / mydata_train_12.__len__()))
            model.eval()
            auc_label = []
            auc_out = []
            test_acc = 0.0
            with torch.no_grad():

                for data in dataloader_test_12:
                    input, label = data
                    input = input.cuda()
                    label = label.cuda()
                    out = model(input)
                    out = out.cuda()

                    result_loss = loss(out, label)
                    result_loss = result_loss.cuda()

                    auc_label.append(label.cpu().numpy())
                    auc_out.append(out.cpu())

                    test_acc = test_acc + ((out.argmax(1) == label).sum())
                #
                # print("test")

                auc_number, aupr ,pr,rec= caculateAUC(auc_out, auc_label)
                print("acc:{},auc:{},auprc:{},precision:{},recall:{},f1:{}".format(float(test_acc / mydata_test_12.__len__()), auc_number,aupr,pr,rec,2*(pr*rec)/(pr+rec)))

                all_acc = all_acc + float(test_acc / mydata_test_12.__len__())
                all_auc = all_auc + auc_number

            print(all_acc)
            print(all_auc)


    return 0


Train_Test(ftumor='TrainingData/CancerTrain.txt', n=10, feval_tumor='TrainingData/CancerEval.txt',
           feval_normal='TrainingData/ControlEval.txt', STEPs=20000, rate=0.33, fnormal='TrainingData/ControlTrain.txt')