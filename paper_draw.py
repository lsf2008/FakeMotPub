import matplotlib.pyplot as plt
import numpy  as np

flg = 'token_draw'
if flg =='token_draw':
    blk = [32, 40, 48, 56, 64]
    auc = [0.849, 0.839, 0.879, 0.838,0.824]
    plt.plot(blk, auc,  marker='*', color='r', markersize=10,
             markeredgecolor='b',markerfacecolor='b')
    # plt.title('Comparison with different')
    plt.xlabel('Token size')
    plt.ylabel('AUC')
    plt.grid()
    plt.tight_layout()
    plt.show()

if flg == 'rec_layers':
    pass

if flg == 'var_layers':
    pass

if flg == '':
    pass