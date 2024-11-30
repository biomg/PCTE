import os
import pathlib
import re

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from model import PCTE
from sklearn import metrics
from sklearn.metrics import precision_score, recall_score

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
    def __init__(self, x):
        self.x = x


    def __getitem__(self, index):
        gene = torch.from_numpy(self.x[index])

        # gene=gene.float()

        return gene

    def __len__(self):
        return self.x.shape[0]



def Train_Test(ftumor, fnormal, feval_tumor, feval_normal, rate=0.33, n=1, STEPs=10000, dir_prefix=curPath + '/tmp'):
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


        Train_data=np.load("Train_data.npy")
        Train_label=np.load("Train_label.npy")
        yes=[]
        no=[]
        explain_yes_result=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        explain_no_result=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        for (i,l) in enumerate(Train_label):

             if l==1:
                 yes.append(Train_data[i])
             else:
                 no.append(Train_data[i])
        yes=np.array(yes)
        no=np.array(no)
        print(yes.shape)
        print(no.shape)
        mydata_train_12 = Mydata(yes)
        dataloader_train_12 = DataLoader(dataset=mydata_train_12, batch_size=200, shuffle=True)
        mydata_test_12 = Mydata(no)
        dataloader_test_12 = DataLoader(dataset=mydata_test_12, batch_size=200, shuffle=True)
        model = PCTE()
        model = model.cuda()
        model.load_state_dict(torch.load('model_weights.pth'))

        # 如果你需要将模型设置为评估模式（例如禁用dropout等）
        model.eval()
        s=torch.nn.Softmax()
        all_yes_out=[]
        all_no_out=[]
        with torch.no_grad():

                for input in dataloader_train_12:


                     input = input.cuda()

                     out = s(model(input))



                     for o in out.cpu():
                      all_yes_out.append(o[1].item())
                for input in dataloader_test_12:


                     input = input.cuda()

                     out = s(model(input))



                     for o in out.cpu():
                      all_no_out.append(o[0].item())
        all_yes_out=np.array(all_yes_out)
        for acid_id in range(0,20):
            yes_a=yes

            yes_a[yes_a == acid_id] = 20
            mydata_train_12 = Mydata(yes_a)
            dataloader_train_12 = DataLoader(dataset=mydata_train_12, batch_size=200, shuffle=True)
            acid_out = []
            with torch.no_grad():

                for input in dataloader_train_12:

                    input = input.cuda()

                    out = s(model(input))

                    for o in out.cpu():
                        acid_out.append(o[1].item())
            acid_out = np.array(acid_out)
            acid_out=acid_out.sum()
            explain_yes_result[acid_id]=(all_yes_out.sum()-acid_out.sum())/all_yes_out.shape[0]
        print(explain_yes_result)
        all_no_out=np.array(all_no_out)
        for acid_id in range(0,20):
            no_a=no

            no_a[no_a == acid_id] = 20
            mydata_train_12 = Mydata(no_a)
            dataloader_train_12 = DataLoader(dataset=mydata_train_12, batch_size=200, shuffle=True)
            acid_out = []
            with torch.no_grad():

                for input in dataloader_train_12:

                    input = input.cuda()

                    out = s(model(input))

                    for o in out.cpu():
                        acid_out.append(o[0].item())
            acid_out = np.array(acid_out)
            acid_out=acid_out.sum()
            explain_no_result[acid_id]=(all_no_out.sum()-acid_out.sum())/all_no_out.shape[0]
        print(explain_no_result)
        acid = all_dict = ['W', 'F', 'G', 'A', 'V', 'I', 'L', 'M', 'P', 'Y', 'S', 'T', 'N', 'Q', 'C', 'K', 'R', 'H',
                           'D', 'E']
        yes = np.array( explain_yes_result)
        no = np.array(explain_no_result)

        # 获取排序后的索引
        indices = np.argsort(yes)
        yes = []
        for id in indices:
            yes.append(acid[id])
        print(yes)
        indices = np.argsort(no)

        no = []
        for id in indices:
            no.append(acid[id])
        print(no)



    return 0


Train_Test(ftumor='TrainingData/CancerTrain.txt', n=1, feval_tumor='TrainingData/CancerEval.txt',
           feval_normal='TrainingData/ControlEval.txt', STEPs=20000, rate=0.33, fnormal='TrainingData/ControlTrain.txt')