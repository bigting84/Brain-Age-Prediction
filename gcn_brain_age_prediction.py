#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 14:14:55 2019

@author: mliu
"""

import pandas as pd
from scipy.sparse import csr_matrix
import numpy as np
import matplotlib.pyplot as plt
from lib import models, graph, coarsening
import config
from lib.utils import train_val_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

age_df = pd.read_excel('./resources/neonates_age_all_clean.xlsx')
age_df = age_df.loc[:,['ID','scan age']]

df_values_list = [pd.read_table('./data/pre_' + str(ID) + '_features_vertices.txt', header=None).values for ID in age_df['ID']]

df_values = np.stack(df_values_list,axis=0)
df_values = df_values[:,:,[0,1,2]].astype('float32')
n_features = df_values.shape[2]
    
graph_edges = pd.read_table('resources/edges1k.txt',names =['node1','node2'],encoding='UTF-16')
#num_nodes = graph_edges.iloc[:,0].max()+1
num_nodes=graph_edges.stack().max()

links = graph_edges.values
gg = np.random.permutation(links[:,1])
graph_edges['node3']=pd.DataFrame(gg)
adj_matrix = csr_matrix((np.ones(graph_edges.shape[0]),
            (graph_edges['node1'].values-1,
            graph_edges['node2'].values-1)),
           shape=(num_nodes,num_nodes))

r,c = adj_matrix.nonzero()
adj_matrix[c,r] = adj_matrix[r,c]

cfg = coarsening.coarsen(adj_matrix, levels=9, self_connections=False)
graphs = cfg[0]
perm = cfg[1]


df_values = df_values[:, 0:num_nodes, :]


data = coarsening.perm_data(df_values[:,:,:], perm)
data = data.astype('float32')

L = [graph.laplacian(A, normalized=True).astype('float32') for A in graphs]

y = age_df['scan age'].values
# kf = KFold(n_splits=5,shuffle=True,random_state=98)
# kf_split = kf.split(X=data,y=y)

test_pred_all, test_labels_all, fold,  IDall = [],[],[],[]

params = vars(config.args)

# for train_ind, test_ind in kf_split:
for i in range(0, 5):
    train_ind, valid_ind, test_ind = train_val_test_split(fold=i, num=len(data))
    print('data shape',data.shape)
    train_data = data[train_ind,:,:]
    train_data = train_data.reshape([train_data.shape[0],-1])
    scaler = StandardScaler()
    train_data_trans = scaler.fit_transform(train_data)
    train_data_trans = train_data_trans.reshape([train_data.shape[0],-1,n_features])
    train_labels = y[train_ind].astype('float32')

    valid_data = data[valid_ind,:,:]
    valid_data = valid_data.reshape([valid_data.shape[0],-1])
    valid_data_trans = scaler.transform(valid_data)
    valid_data_trans = valid_data_trans.reshape([valid_data.shape[0],-1,n_features])
    valid_labels = y[valid_ind].astype('float32')

    test_data = data[test_ind,:,:]
    test_data = test_data.reshape([test_data.shape[0],-1])
    test_data_trans = scaler.transform(test_data)
    test_data_trans = test_data_trans.reshape([test_data.shape[0],-1,n_features])
    test_labels = y[test_ind].astype('float32')

    mean_predict = np.tile(train_labels.mean(), valid_labels.shape)
    print('baseline MSE: ', mean_squared_error(valid_labels,mean_predict))

    model = models.cgcnn(L, **params)
    model.fit(train_data=train_data_trans,train_labels=train_labels,val_data=valid_data_trans,val_labels=valid_labels)

    y_pred = model.predict(data=test_data_trans)

    plt.scatter(test_labels, y_pred)

    test_pred_all.append(y_pred)
    test_labels_all.append(test_labels)
    IDall.append(age_df.ID[test_ind].tolist())
    fold.append([i]*len(y_pred))

    break

flatten = lambda z: [x for y in z for x in y]
pred_array = flatten(test_pred_all)
true_array = flatten(test_labels_all)
IDall = flatten(IDall)
fold = flatten(fold)

df_out = pd.DataFrame({'ID':IDall, 'true':true_array,'pred':pred_array,'fold':fold})

output_fn = 'predicted_age'
df_out.to_csv(output_fn + '.csv')

plt.xlim([20,45])
plt.ylim([20,45])
plt.xlabel('true age (weeks)')
plt.ylabel('predicted age (weeks)')
plt.savefig('test.png')
