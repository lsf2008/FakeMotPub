import matplotlib.pyplot as plt
import numpy  as np

flg = 'token_draw+coef_cmb'

from qbstyles import mpl_style
def drw_plot(blk, auc, xlb='Token size', ylb='AUC', maxsize = None):

    # plt.style.use('seaborn-muted')
    plt.plot(blk, auc, marker='*', color='r', markersize=10,
             markeredgecolor='b', markerfacecolor='b')
    # plt.title('Comparison with different')
    plt.xlabel(xlb, fontsize=13)
    plt.ylabel(ylb, fontsize=13)

    plt.grid()
    # plt.gca().margins(x=5)
    # plt.gcf().canvas.draw()
    if maxsize:
        m = 0.2
        N = len(blk)
        s = maxsize / plt.gcf().dpi * N + 2 * m
        margin = m / plt.gcf().get_size_inches()[0]

        plt.gcf().subplots_adjust(left=margin, right=1. - margin)
        plt.gcf().set_size_inches(s, plt.gcf().get_size_inches()[1])
    plt.tight_layout()
    return plt
    # plt.show()
if flg =='token_draw+coef_cmb':
    blk = [24, 32, 40, 48, 56, 64]
    auc = [0.945, 0.946, 0.921, 0.961, 0.921, 0.886]
    mpl_style(dark=False)
    plt.figure(figsize=(10,3))
    plt.subplot(1,2,1)
    # plt.figure(figsize=(5,3))
    drw_plot(blk, auc, 'Token size', 'AUC', maxsize=None)
    plt.grid()

    plt.subplot(1, 2, 2)
    alpha = [16, 32, 64, 128, 256]
    coefAUC = [0.937, 0.913, 0.927, 0.930, 0.928]
    # plt.figure(figsize=(5, 3))
    drw_plot(alpha, coefAUC, 'Neurons of hidden layer', 'AUC', None)
    plt.grid()
    plt.tight_layout()
    plt.show()

if flg =='coef_cmb':
    alpha = [1, 5, 10, 15, 20, 25, 30, 35, 40]
    coefAUC = [0.829, 0.872, 0.873, 0.876, 0.875, 0.879, 0.868, 0.868, 0.873]
    plt.figure(figsize=(5, 3))
    drw_plot(alpha, coefAUC, 'Coefficients', 'AUC',None)
if flg == 'rec_layers':
    pass

if flg == 'var_layers':
    pass

if flg == '':
    pass