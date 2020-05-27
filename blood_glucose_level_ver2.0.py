"""
1.import modules
  モジュールのインポート
"""

%load_ext autoreload

%autoreload 2
import time

import pandas
import argparse

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions
from chainer import Link, Chain, ChainList

from chainer.datasets import tuple_dataset
from chainer.dataset import convert, concat_examples

from chainer import datasets, iterators, optimizers, serializers
from chainer import Function, report, training, utils, Variable

import numpy
import os
import glob
import sys
from pathlib import Path
import matplotlib.pyplot as plt

from chainer import Sequential
import easy_chainer

"""
2.Define of structure
  構造の定義
  MLP_01 : dropout, Batch_normalization = None
  MLP_02 : dropout, Batch_normalization
"""


class MLP_01(chainer.Chain):

    def __init__(self, n_units, n_out):
        super(MLP_01, self).__init__()
        with self.init_scope():
            # the size of the inputs to each layer will be inferred
            self.l1 = L.Linear(None, n_units)  # n_in -> n_units
            self.l2 = L.Linear(None, n_units)  # n_units -> n_units
            self.l3 = L.Linear(None, n_out)  # n_units -> n_out

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        return self.l3(h2)


# 4層NN(入力→中間，中間→出力)
class MLP_02(chainer.Chain):
    """
    モデルの実装
    """

    def __init__(self, n_units, n_out, train=True, drop_out_ratio=0.3):
        super(MLP_02, self).__init__()
        with self.init_scope():
            # the size of the inputs to each layer will be inferred
            self.l1 = L.Linear(None, n_units)  # n_in -> n_units
            self.l2 = L.Linear(None, n_units)  # n_units -> n_units
            self.l3 = L.Linear(None, n_out)  # n_units -> n_out

        # 学習の場合：True
        self.__train = train
        # drop outの実施有無
        self.__drop_out = True
        # drop outの比率
        self.drop_out_ratio = drop_out_ratio

    def __call__(self, x):
        drop_out = self.__train and self.__drop_out
        h1 = F.dropout(F.relu(self.l1(x)), ratio=self.drop_out_ratio)
        h2 = F.dropout(F.relu(self.l2(h1)), ratio=self.drop_out_ratio)
        return self.l3(h2)

    # 学習の場合；True
    def __get_train(self):
        return self.__train

    def __set_train(self, train):
        self.__train = train

    train = property(__get_train, __set_train)

    # Dropoutを使用する場合：True
    def __get_drop_out(self):
        return self.__drop_out

    def __set_drop_out(self, drop_out):
        '''
        drop outフラグの設定
        '''
        self.__drop_out = drop_out

    drop_out = property(__get_drop_out, __set_drop_out)


def parse_device(args):
    gpu = None
    if args.gpu is not None:
        gpu = args.gpu
    elif re.match(r'(-|\+|)[0-9]+$', args.device):
        gpu = int(args.device)

    if gpu is not None:
        if gpu < 0:
            return chainer.get_device(numpy)
        else:
            import cupy
            return chainer.get_device((cupy, gpu))

    return chainer.get_device(args.device)

"""
3. Import datasets
  データセット読み込み
  　FBGセンサから取得した脈波データ
  　参照血糖値
"""

# 1,000datas
data, teach = easy_chainer.load_Data("C:/Users/Owner/Desktop/Normalized/val/val_ebina_day1.xlsx")
data = data.astype(numpy.float32)
teach = teach
print(teach)
print(teach.shape)

# 回帰させるときに必要（分類はint型）
teach = teach.astype(numpy.float32)

id_all = numpy.arange(1, len(teach) + 1, 1).astype(numpy.int32) - 1
# print(id_all)
numpy.random.seed(11)
id_train = numpy.random.choice(id_all, 400, replace=False) #重複なし
print(id_train)
id_test = numpy.delete(id_all, id_train)
print(id_test)

# train, test データをExcelに保存するスクリプト
# このスクリプトを追加する

teach_train = teach[id_train]
df_teach = pandas.DataFrame(teach_train, columns=['train'])
df_idtrain = pandas.DataFrame(id_train, columns=['id_train'])
df_train = pandas.concat([df_idtrain, df_teach], axis=1)
# print(df_train)

teach_test = teach[id_test]
df_id_test = pandas.DataFrame(teach_test, columns=['test'])
df_idtest = pandas.DataFrame(id_test, columns=['id_test'])
df_test = pandas.concat([df_idtest, df_id_test], axis=1)
# print(df_test)

df = pandas.concat([df_train, df_test], axis=1)
df.to_excel("C:/Users/Owner/Desktop/Normalized/blood_glucose_teach_20191107_01.xlsx")
# print(df)

m = 11

x_train, y_train = data[:, 0:m], teach[0:m]
x_test, y_test = data[:, m:22], teach[m:22]

print(y_train.shape, y_train[1])
print(y_test.shape, y_test[1])

"""
4. Separate datas
 訓練と検証に分ける
"""

x_train, y_train = data[:, id_train], teach[id_train]
x_test, y_test = data[:, id_test], teach[id_test]

"""
5. Generate iterator
 イテレータの生成
"""
train = tuple_dataset.TupleDataset(x_train.T, y_train.reshape(-1,1 ))
test = tuple_dataset.TupleDataset(x_test.T, y_test.reshape(-1, 1))

train_iter = chainer.iterators.SerialIterator(train, 3, repeat=True, shuffle=False)
test_iter = chainer.iterators.SerialIterator(test, 3, repeat=False, shuffle=False)

"""
6. Define of models
  モデル定義
"""
# model determination
net = MLP_02(1000,1)
model = L.Classifier(net,
                     lossfun=F.mean_squared_error,
                     accfun = F.r2_score)
model.compute_accuracy = False

# define optimizer
# 最適化方法
#optimizer = chainer.optimizers.SGD()
optimizer = chainer.optimizers.Adam()
optimizer.setup(model)

"""
7. Trainer
define updater
"""
updater = training.updaters.StandardUpdater(train_iter, optimizer)
trainer = training.Trainer(updater, (500, 'epoch'), out="Result2018_oono/%s" % time.strftime("%Y%m%d%H%M%S"))

# Evaluate the model with the test dataset for each epoch
trainer.extend(extensions.Evaluator(test_iter, model))

# Dump a computational graph from 'loss' variable at the first iteration
# The "main" refers to the target link of the "main" optimizer.
trainer.extend(extensions.dump_graph('main/loss'))

# Take a snapshot for each specified epoch
# frequency = args.epoch if args.frequency == -1 else max(1, args.frequency)
# trainer.extend(extensions.snapshot(), trigger=(frequency, 'epoch'))

# Write a log of evaluation statistics for each epoch
trainer.extend(extensions.LogReport())

# Save two plot images to the result dir
trainer.extend(
    extensions.PlotReport(['main/loss', 'validation/main/loss'],
                          'epoch', file_name='loss.png'))
trainer.extend(
    extensions.PlotReport(
        ['main/accuracy', 'validation/main/accuracy'],
        'epoch', file_name='accuracy.png'))

# Print selected entries of the log to stdout
# Here "main" refers to the target link of the "main" optimizer again, and
# "validation" refers to the default name of the Evaluator extension.
# Entries other than 'epoch' are reported by the Classifier link, called by
# either the updater or the evaluator.
trainer.extend(extensions.PrintReport(
    ['epoch', 'main/loss', 'validation/main/loss',
     'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))

# Print a progress bar to stdout
trainer.extend(extensions.ProgressBar())

# if args.resume:
#     # Resume from a snapshot
#     chainer.serializers.load_npz(args.resume, trainer)

"""
訓練実行！
"""
trainer.run()

"""
8.Validation
"""

# 検証（訓練データ）
# train_iter.reset()

train_batch = train_iter.next()
train_spc, train_ref = concat_examples(train_batch)  # Test Dataset
for i in range(3):
    cal_ref = train_ref[i]
    cal_pred = net(train_spc[i].reshape(1, -1)).data
    print("Ref: %d, Pred: %d " % (cal_ref, cal_pred))


# 検証（テストデータ）
# test_iter.reset()

test_batch = test_iter.next()
test_spc, test_ref = concat_examples(test_batch)  # Test Dataset
for i in range(3):
    ref = test_ref[i]
    pred = net(test_spc[i].reshape(1, -1)).data
    print("Ref: %d, Pred: %d " % (ref, pred))
