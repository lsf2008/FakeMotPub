import matplotlib.pyplot as plt
import numpy  as np

flg = 'token_draw'
if flg =='token_draw':
    blk = [32, 40, 48, 56, 64]
    auc = [0.824, 0.866, 0.879, 0.838,0.824]
    plt.plot(blk, auc)
    # plt.title('Comparison with different')
    plt.xlabel('token size')
    plt.ylabel('AUC')
    plt.grid()

if flg == 'rec_layers':
    pass

if flg == 'var_layers':
    pass

if flg == '':
    pass