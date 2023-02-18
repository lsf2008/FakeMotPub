#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   OneClsBase.py    
@Contact :   lishifeng2007@gmail.com
@License :   (C)Copyright 2017-2018, Liugroup-NLPR-CASIA

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/10/11 20:52   shifengLi     1.0         None
'''

# import lib
from torch.nn import Module
import torch
class OneClassBase(Module):
    # show the class property
    def __repr__(self):
        # type: () -> str
        """
        String representation
        """
        good_old = super(OneClassBase, self).__repr__()
        addition = 'Total number of parameters: {:,}'.format(self.n_parameters)

        return good_old + '\n' + addition

    def cmpDis(self, a, b):
        '''compute distances between set a and b

        Parameters
        ----------
        a : [2d tensor samples1*dim]
            data set a
        b : [2d tensor samples2*dim]
            dataset b

        Returns
        -------
        [2d tensor samples1*samples2]
            distance matrix
        '''

        # if isinstance(a, np.ndarray):
        #     a = torch.as_tensor(a)
        # if isinstance(b, np.ndarray):
        #     b = torch.as_tensor(b)
        # if device is not None:
        #     a, b = a.to(device), b.to(device)

        sq_a = a ** 2
        sum_sqa = torch.sum(sq_a, dim=1).unsqueeze(1)
        sq_b = b ** 2
        sum_sqb = torch.sum(sq_b, dim=1).unsqueeze(0)
        bt = b.t()
        # bt = torch.tensor(bt, dtype=a.dtype)
        return sum_sqa + sum_sqb - 2 * a.mm(bt)

    def obtClusterLbs(self, dis):
        '''find the cluster labels of dt
        Parameters
        ----------
        dis : [tensor/ndarray samples1*samples2]
            distances, one row contains the distances between
            one sample in a to the samples in b
        clster : [tensor of clusters* dimension]
            cluster centers array

        Returns
        -------
        lb:[tensor ]
            cluster label for each data
        dis:[tensor]
            and the corresponding minmum distance
        '''
        # distance between data and clusters
        # dis = cmpDis(dt, clster, device)
        # if dis.isnan().any():
        #     print(f'dis is nan {dis.cpu()}')
        md, lb = torch.min(dis, dim=1)
        return lb, md

    def cmpSkddRDis(self, dismin, lbs, R):
        '''
        compute difference between sample and R in each center
        Parameters
        ----------
        dismin: (b,1) sample distances in each center
        lbs: (b,1) center label for each sample
        R: (k,) radius of each cluster
        Returns
        -------
        scorek: (b,1) square distance between sample and R in each center.
        '''
        # dismin = self.dis_kmean
        # lbs = self.labels_
        scorek = torch.zeros_like(dismin)
        k = torch.unique(lbs)
        # R = torch.tensor((k,), dtype=torch.float)
        for i in k:
            # flg = lbs==i
            # # 该聚类中元素为0
            # if torch.sum(flg==True)==0:
            #     continue

            disk = dismin[lbs == i]
            rk = R[i]
            scorek[lbs == i] = (disk - rk ** 2)

            # lossk = rk**2+(1/nu)*torch.mean(torch.max(torch.zeros_like(scorek), scorek))
        return scorek