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
def normalizeDicScore(scoreDic):
    '''scoreDIc is a dictionary, like:
    {'classScores': [tensor(0.5000), tensor(0.5000), tensor(0.5000), tensor(0.6000), tensor(0.6000), tensor(0.6000)], 'motScores': [tensor(0.1000), tensor(0.1000), tensor(0.1000), tensor(0.3000), tensor(0.3000), tensor(0.3000)], 'aeScores': [tensor(0.7000), tensor(0.7000), tensor(0.7000), tensor(0.1000), tensor(0.1000), tensor(0.1000)]}, [tensor(1), tensor(1), tensor(1), tensor(0), tensor(0), tensor(0)]

    Parameters
    ----------
    scoreDic : dictorary-like object. The output of obtAllScoresFrmDicOutputs. It is a dictionary. The keys are the names of the inputs

    Returns
    -------
    dictionary
    '''
    sDic ={}
    for key, vaules in scoreDic.items():  # keys is string, vaules is float tensor.  vaules is float tensor.  v
        vaules = torch.tensor(vaules)
        maxs = torch.max((vaules) )
        mins = torch.min((vaules))
        sDic[key]=(vaules - mins) / (maxs - mins + 1e-7)
    return sDic

def obtAllScoresFrmDicOutputs(dicList):
    '''
    dicList is dictionary, maybe it is like:
    dicList=[{'classScores': 0.5,
                    'motScores': 0.6,
                    'aeScores': 0.7,
                    'label': 1},
                    {'classScores': 0.6,
                    'motScores': 0.5,
                    'aeScores': 0.6,
                    'label': 0}]
    :param outputs:
    :return:
    '''
    keys = list(dicList[0].keys())

    # init list
    lst = [[] for i in range(len(keys))]

    for x in dicList:
        # print(x)
        cnt = 0
        for key, value in x.items():
            lst[cnt].extend(value)
            cnt += 1

    resDic = {}

    for i in range(len(lst) - 1):
        resDic[keys[i]] = lst[i]

    resDic = normalizeDicScore(resDic)

    return resDic, lst[-1]

def obtMotAllScoresFrmOutputs(outputs):
    '''
    # obtain all scores and corresponding y
    :param outputs: validation or test end outputs
    :return: dict{scores:}, label:list...

    '''
    softScores = []
    motScores = []
    y_true = []
    for i, out in enumerate(outputs):
        motScore, aeScore, y = out
        softScores.extend(aeScore)
        motScores.extend(motScore)
        y_true.extend(y)

    # normalize
    softScores = torch.tensor(softScores)
    softScores = utils.normalize(softScores)

    motScores = torch.tensor(motScores)
    motScores = utils.normalize(motScores)
    scoreDic = {'predScores': softScores,
                'motScores': motScores}

    return scoreDic, y_true

def obtAllScoresFrmOutputs(outputs):
    '''
    # obtain all scores and corresponding y
    :param outputs: validation or test end outputs
    :return: dict{scores:}, label:list...

    '''
    aeScores = []
    motScores = []
    y_true = []
    for i, out in enumerate(outputs):
        aeScore, motScore, y = out
        aeScores.extend(aeScore)
        motScores.extend(motScore)
        y_true.extend(y)

    # normalize
    aeScores = torch.tensor(aeScores)
    aeScores = utils.normalize(aeScores)

    motScores = torch.tensor(motScores)
    motScores = utils.normalize(motScores)
    scoreDic = {'aeScores': aeScores,
                'motScores': motScores}

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
                cmbScores[p] =torch.stack(list(scoreDic.values())).t()@torch.tensor(p, dtype=torch.float).unsqueeze(1)

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
def filterCrops(x, thr=0.01):
    '''
    remove background
    Parameters
    ----------
    x:      n, c, t, h, w
    thr:    threshold
    Returns
    -------
    '''

    x = x.reshape((-1, *x.shape[2:]))
    bt = x.shape[0]
    x1 = x > 0.001

    xn = []

    v = torch.prod(torch.tensor(x1.shape[1:])) * thr
    for i in range(bt):
        t = x[i]
        # print(torch.sum(x1[i,:,:,:,:]) )
        if torch.sum(x1[i]) > v:
            xn.append(t)
    return torch.stack(xn)
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
            # y_true = [i.detach().cpu() for i in y_true]
            # v = v.detach().cpu()
            # val_roc = roc_auc_score(y_true, v)
            val_roc = calAUC(v, y_true)
            # print()
            # print(f'current auc:{val_roc}'.center(100, '-'))
            if res['maxAuc'] < val_roc:
                res['maxAuc'] = val_roc
                res['coef'] = str(k)
                res['label'] = y_true
                res['score'] = v
                res['epoch'] = epoch
    return res
if __name__=='__main__':
    x = torch.rand((120, 3, 8, 64, 64))
    y = filterCrops(x, 0.1)
    print(y.shape)
    # obj = FilterCrops()
    # print(obj.shape)

