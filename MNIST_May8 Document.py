import argparse

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions


from chainer.datasets import tuple_dataset
from chainer.dataset import convert, concat_examples
from pylab import *

import pandas
import numpy



class MLP(chainer.Chain):

    def __init__(self, n_units, n_out):
        super(MLP, self).__init__()
        with self.init_scope():
            # the size of the inputs to each layer will be inferred
            self.l1 = L.Linear(None, n_units)  # n_in -> n_units
            self.l2 = L.Linear(None, n_units)  # n_units -> n_units
            self.l3 = L.Linear(None, n_out)  # n_units -> n_out

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        return self.l3(h2)


net = MLP(1000, 10)
model = L.Classifier(net)
# optimizer = chainer.optimizers.Adam(alpha=0.001, beta1=0.9, beta2=0.999, eps=1e-08)
optimizer = chainer.optimizers.SGD()
optimizer.setup(model)

train, test = chainer.datasets.get_mnist()
train_iter = chainer.iterators.SerialIterator(train, 1000)
test_iter = chainer.iterators.SerialIterator(test, 1000, repeat=False, shuffle=False)

train_iter.reset()
tmp = train_iter.next()[0]

# print(len(tmp[0]))
# %matplotlib inline
# from pylab import *
# imshow(tmp[0].reshape(28, 28))
# print(tmp[0])

# イタレータの初期化です
train_iter.reset()
test_iter.reset()

# ----------------------------

# 学習回数
# iteration 回数(パラメータの更新回数)でも指定できます
epoch = 10

# GPU を使うか否か;
gpu = -1


# 学習sequence の初期化
updater = training.updaters.StandardUpdater(
    train_iter, optimizer, device=gpu)
trainer = training.Trainer(updater, (epoch, 'epoch'), out="./Result20180612")

# 検証の設定です: 訓練したモデルに対して, 検証データを代入して評価します.
# 毎epoch (ないし, iteration)で計算されます
trainer.extend(extensions.Evaluator(test_iter, model, device=gpu))

# 使ったネットワークを画像化するための, dot 形式のファイルを出力する設定です.
# graphviz というグラフ作成
# ここでいうグラフは, 折れ線グラフとか散布図グラフの"グラフ"でなく,
# ノードやエッジの集合で表す"グラフ"です.
trainer.extend(extensions.dump_graph('main/loss'))


# Write a log of evaluation statistics for each epoch
trainer.extend(extensions.LogReport())

# loss(損失) Plotの設定
trainer.extend(
    extensions.PlotReport(['main/loss', 'validation/main/loss'],
                          'epoch', file_name='loss.png'))

# accuracy(分類精度) Plotの設定
trainer.extend(
    extensions.PlotReport(
        ['main/accuracy', 'validation/main/accuracy'],
        'epoch', file_name='accuracy.png'))

# 標準出力で, どの項目を出力させるかの設定です.
trainer.extend(extensions.PrintReport(
    ['epoch', 'main/loss', 'validation/main/loss',
     'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))

trainer.run()

test_iter.reset()

test_batch = test_iter.next()
test_spc, test_ref = concat_examples(test_batch)  # Test Dataset
for i in range(10):
    ref = test_ref[i]
    pred = numpy.argmax(net(test_spc[i].reshape(1, -1)).data)
    print("Ref: %d, Pred: %d %s" % (ref, pred, ref == pred))
