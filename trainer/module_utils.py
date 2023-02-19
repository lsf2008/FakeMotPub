import torch
import itertools
import utils
from sklearn.metrics import roc_auc_score
def obtAScoresFrmOutputs(outputs):
    '''
    # obtain all scores and corresponding y
    :param outputs: validation or test end outputs
    :return: dict{scores:}, label:list...

    '''
    aeScores = []
    y_true = []
    for i, out in enumerate(outputs):
        aeScore, y = out
        aeScores.extend(aeScore)
        y_true.extend(y)

    # normalize
    aeScores = torch.tensor(aeScores)
    aeScores = utils.normalize(aeScores)
    scoreDic = {'aeScores': aeScores}

    return scoreDic, y_true


def cmbScoreWght(scoreDic, weight=None):
    '''
    combine the scores with weight
    :param scoreDic: score dic {'aeScore': tensor}
    :param weight: list, lenght = len(scoreDic)
    :return: tensor of combined score
    '''
    itms = len(scoreDic)

    cmbScores = {}
    if itms == 1:
        cmbScores[1] = scoreDic[list(scoreDic.keys())[0]]
    else:
        if weight == None:
            cmbScore = torch.zeros_like(list(scoreDic.values())[0])
            for i, k, v in enumerate(scoreDic.items()):
                cmbScore += v
            cmbScores[1] = cmbScore
        else:
            for p in itertools.combinations(weight, itms):
                cmbScore = torch.zeros_like(list(scoreDic.values())[0])
                for i, k, v in enumerate(scoreDic.items()):
                    cmbScore += v * p[i]
                cmbScores[p] = cmbScore
    return cmbScores

def calAUC(prob, labels):
    f = list(zip(prob, labels))
    rank = [values2 for values1, values2 in sorted(f, key=lambda x: x[0])]
    rankList = [i + 1 for i in range(len(rank)) if rank[i] == 1]
    posNum = 0
    negNum = 0
    for i in range(len(labels)):
        if (labels[i] == 1):
            posNum += 1
        else:
            negNum += 1
    auc = 0
    auc = (sum(rankList) - (posNum * (posNum + 1)) / 2) / (posNum * negNum)
    # print(auc)
    return auc


def cmpCmbAUCWght(scoreDic, weight, y_true, res, epoch):
    '''
    compute AUC for one weight and score
    :param scoreDic: score dic {'aeScore': tensor}
    :param weight: list, lenght = len(scoreDic)
    :param y_true: groud truth
    :param res: dictory to store results
    :return: dictory
    '''
    cmbScores = cmbScoreWght(scoreDic, weight)

    if sum(y_true)==len(y_true):
        y_true[0]=0
    if sum(y_true)==0:
        y_true[0]=1
    if epoch:
        for k, v in cmbScores.items():
            # val_roc = roc_auc_score(y_true, v)
            val_roc = calAUC(v, y_true)
            print('/n')
            print(f'current auc:{val_roc}'.center(100, '-'))

            if res['maxAuc'] < val_roc:
                res['maxAuc'] = val_roc
                res['coef'] = k
                res['label'] = y_true
                res['score'] = v
                res['epoch'] = epoch
    return res

