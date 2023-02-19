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
        aeScores.extend(aeScore.detach().cpu().numpy())
        y_true.extend(y.detach().cpu().numpy())

    # normalize
    aeScores = torch.tensor(aeScores)
    aeScores = utils.normalize(aeScores)
    scoreDic = {'aeScores': (aeScores)}

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


def cmpCmbAUCWght(scoreDic, weight, y_true, res):
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
    if res['epoch']:
        for k, v in cmbScores.items():
            val_roc = roc_auc_score(y_true, v)

            if res['maxAuc'] < val_roc:
                res['maxAuc'] = val_roc
                res['coef'] = k
                res['label'] = y_true
                res['score'] = v
    return res

