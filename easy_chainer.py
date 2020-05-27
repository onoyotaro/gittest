#! coding: utf-8
import chainer
from chainer import training  # multi-task updaterで使ってる


class Architecture(chainer.Chain):
    """ Architecture(net_list)
    list 型で記述した ネットワーク構造をchainer.Chain に変換します.
    Chainer v5.0.0 P.71 を参考にしました.
    """

    def __init__(self, net_list=None):
        super(Architecture, self).__init__()
        with self.init_scope():
            for n in net_list:
                if not n[0].startswith('_'):
                    setattr(self, n[0], n[1])
        self.layers = net_list

    def __call__(self, x):
        # x = x.reshape(x.shape[0], 1, -1, 1)
        for n, f in self.layers:
            if not n.startswith('_'):
                x = getattr(self, n)(x)
            else:
                x = f(x)
        return x


class Logger:
    """ 標準出力 と logging """

    def __init__(self, result_dir=None):
        self.result_dir = result_dir

    def log(self, _tag="", _msg=""):
        FORM = "%s - [%10s] %s"
        print(FORM % (now(), _tag, _msg))
        if self.result_dir:
            with open("%s/history.txt" % (self.result_dir), "a") as f:
                f.write(FORM % (now(), _tag, _msg) + "\n")


def Args():
    """ 引数の管理 """
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--cnf", default="_json/configure002.json")

    return p.parse_args()


def now(fmt="%Y/%m/%d.%H:%M:%S"):
    """ 現在時刻の取得 """
    import time
    return time.strftime(fmt)


def read_cnf(_file="./_json/configure001.json", q=""):
    import json
    import collections
    decoder = json.JSONDecoder(object_pairs_hook=collections.OrderedDict)

    with open(_file, "r") as f:
        _json = f.read()
    if q:
        decode_json = decoder.decode(_json)[q]
    else:
        decode_json = decoder.decode(_json)
    return decode_json


def load_Data(fname="", output_all=0, no_teach=0):
    """ Load Data """
    import pandas
    import numpy
    import os

    file_type = os.path.split(fname)[1]
    xls = pandas.read_excel(fname)
    column = xls.keys().tolist()
    csv_index = xls.index[:-1]
    if no_teach:
        return xls.as_matrix()
    else:
        data = xls.as_matrix()[:-1]
        teach = xls.as_matrix()[-1]
        if output_all:
            return data, teach, column, csv_index
        else:
            return data, teach


def prep_dataset(cnf="_json/configure001.json",
                 result_dir="./",
                 output=0,
                 logger=None,
                 task="classifier"):
    """ dataset の準備 """
    from chainer.datasets import tuple_dataset
    from chainer import iterators, serializers, training, reporter
    import numpy
    import os
    import json

    dataset_cnf = read_cnf(_file=cnf, q="Dataset")

    xls = dataset_cnf["xls"]
    TEST_RATIO = dataset_cnf["test_ratio"]
    SEED = dataset_cnf["seed"]
    logger.log(_tag="Dataset", _msg="Load Data: %s" % xls)

    data, teach = load_Data(xls)
    data = data.astype(numpy.float32)
    numpy.random.seed(SEED)

    logger.log(
        _tag="Dataset", _msg="Separator >> TEST RATIO: %1.1f" % TEST_RATIO)
    logger.log(_tag="Dataset", _msg="Separator >> SEED: %d" % SEED)
    if task == "classifier":
        teach = teach.astype(numpy.int8)
        id_all = numpy.arange(1, len(teach) + 1, 1).astype(numpy.int32) - 1

        if output:
            print(numpy.argmax(teach) + 1)
        teach_bins = numpy.histogram(teach, bins=numpy.max(teach) + 1)[0]
        if output:
            print("teach_bins: ", teach_bins)

        for i in range(int(max(teach) + 1)):
            k = numpy.round(teach_bins[i] * TEST_RATIO)
            tmp0_arr = numpy.where(teach == i)[0]
            try:
                tmp0 = numpy.random.choice(tmp0_arr, int(k), replace=False)
                if i == 0:
                    id_test = tmp0
                else:
                    id_test = numpy.hstack((id_test, tmp0))
            except Exception:
                print("Not teach")

        id_train = numpy.delete(id_all, id_test)

    if task == "regression":
        teach = teach.astype(numpy.float32)
        id_all = numpy.arange(1, len(teach) + 1, 1).astype(numpy.int32) - 1
        k = numpy.round(teach.shape[0] * TEST_RATIO)
        id_test = numpy.random.choice(
            range(teach.shape[0]), int(k), replace=False)
        id_train = numpy.delete(id_all, id_test)

    # 使用したデータセットID の出力
    id_train = id_train.tolist()
    id_test = id_test.tolist()

    import array
    dict_dataset = {
        "id_train": sorted(id_train),
        "id_test": sorted(id_test),
        "xls": xls
    }
    json_dataset = json.dumps(
        dict_dataset, indent=4, sort_keys=1, separators=(",", ": "))
    with open("%s/dataset.json" % (result_dir), "w") as f:
        f.write(json_dataset)

    x_train, y_train = data[:, id_train], teach[id_train]
    x_test, y_test = data[:, id_test], teach[id_test]

    if task == "classifier":
        train = tuple_dataset.TupleDataset(x_train.T, y_train.reshape(-1, ))
        test = tuple_dataset.TupleDataset(x_test.T, y_test.reshape(-1, ))
    if task == "regression":
        train = tuple_dataset.TupleDataset(x_train.T, y_train.reshape(-1, 1))
        test = tuple_dataset.TupleDataset(x_test.T, y_test.reshape(-1, 1))
    return train, test


def prep_iterators(train, test, cnf="./_json/configure001.json", logger=None):
    """ イタレータの生成 """
    from chainer import iterators
    iter_cnf = read_cnf(_file=cnf, q="Iterator")

    logger.log(_tag="Iterator", _msg="create accepted.")
    return (iterators.SerialIterator(
        train, iter_cnf["batch_size"], shuffle=True),
            iterators.SerialIterator(
                test, iter_cnf["batch_size"], repeat=False, shuffle=False))


def prep_optimizer(cnf="_json/configure001.json", logger=None):
    """ 最適化の設定 """
    import chainer
    optimizer_cnf = read_cnf(_file=cnf, q="Optimizer")

    method = list(optimizer_cnf.keys())[0]
    param = optimizer_cnf[list(optimizer_cnf.keys())[0]]
    dict_optimizer = {
        "SGD": chainer.optimizers.SGD(),
        "MomentumSGD": chainer.optimizers.MomentumSGD(),
        "Adam": chainer.optimizers.Adam(),
        "AdaGrad": chainer.optimizers.AdaGrad(),
        "AdaDelta": chainer.optimizers.AdaDelta(),
        "RMSprop": chainer.optimizers.RMSprop(),
        "RMSpropGraves": chainer.optimizers.RMSpropGraves(),
    }
    optimizer = dict_optimizer[method]

    param_keys = list(vars(vars(optimizer)["hyperparam"]).keys())[1:]
    for key in param_keys:
        try:
            setattr(optimizer, key, param[_])
        except Exception:  # set a default value.
            pass
    logger.log(_tag="Optimizer", _msg="Method: %s" % (optimizer.__module__))
    logger.log(_tag="Optimizer", _msg="%s" % (optimizer.hyperparam))
    return optimizer


def prep_updater(_model,
                 _iter,
                 _optimizer,
                 cnf="_json/configure001.json",
                 logger=None):
    " == preparation of optimizer @ easy_chainer.py =="
    from chainer import training

    _optimizer.setup(_model)
    updater_cnf = read_cnf(cnf, q="Updater")

    _gpu = updater_cnf["device"]

    logger.log(_tag="Updater", _msg="Setting accepted.")
    return training.StandardUpdater(
        iterator=_iter, optimizer=_optimizer, device=_gpu)


def prep_Trainer(_model,
                 _updater,
                 result_dir,
                 _test_iter,
                 cnf="./_json/configure001.json",
                 _args=None,
                 logger=None):
    from chainerui.extensions import CommandsExtension
    from chainerui.utils import save_args
    from chainer import training
    from chainer.training import extensions

    trainer_cnf = read_cnf(cnf, q="Trainer")
    _epoch = trainer_cnf["epoch"]
    _device = trainer_cnf["device"]
    _console = trainer_cnf["console"]

    logger.log(_tag="Trainer", _msg="Setting accepted.")

    trainer = training.Trainer(_updater, (_epoch, 'epoch'), out=result_dir)
    trainer.extend(extensions.Evaluator(_test_iter, _model, device=_device))
    trainer.extend(
        extensions.dump_graph(root_name="main/loss", out_name="predictor.dot"))
    trainer.extend(extensions.LogReport())
    trainer.extend(
        extensions.PlotReport(
            ["main/loss", "validation/main/loss"],
            "epoch",
            file_name="loss.png"))
    trainer.extend(
        extensions.PlotReport(
            ["main/accuracy", "validation/main/accuracy"],
            "epoch",
            file_name="accuracy.png"))
    if _console:
        trainer.extend(
            extensions.PrintReport([
                'epoch', 'main/loss', 'validation/main/loss', "main/accuracy",
                "validation/main/accuracy", "elapsed_time"
            ]))

    trainer.extend(extensions.ProgressBar(update_interval=1))
    trainer.extend(extensions.observe_lr())
    trainer.extend(CommandsExtension())
    if _args:
        save_args(args, result_dir)
    return trainer


def save_model(_model, filename, logger=None):
    import chainer

    logger.log(_tag="Serialize", _msg="Model: %s" % (filename))

    chainer.serializers.save_hdf5(filename, obj=_model)


# def save_model_onnx(_model, filename, logger=None):
#     import chainer

#     logger.log(_tag="Serialize", _msg="Model: %s" % (filename))

#     chainer.serializers.save_hdf5(filename, obj=_model)


def validation_classifer(_net, _iter, logger):
    """ 分類問題の validation
    """
    import chainer
    import chainer.functions as F
    import chainer.links as L
    from chainer.dataset import concat_examples

    logger.log(_tag="Report", _msg="Validation of training model.")

    _iter.reset()
    _batch = _iter.next()
    x, t = concat_examples(_batch)

    _model = L.Classifier(_net)

    accuracy = F.accuracy(_net(x), t).data
    loss = _model(x, t).data
    logging(
        _tag="Result", _msg="Loss: %1.2f, Accuracy: %1.2f" % (loss, accuracy))


def validation_regressor(_net,
                         train_iter,
                         test_iter,
                         result_dir,
                         config=None,
                         logger=None):
    """ 回帰問題の validation
    """
    import chainer
    import chainer.functions as F
    import chainer.links as L
    from chainer.dataset import concat_examples
    import matplotlib
    import numpy

    logger.log(_tag="Report", _msg="Validation of training model.")

    _ = [xx.reset() for xx in (train_iter, test_iter)]

    def pred_value(_net, _iter):
        list_y = []
        list_t = []

        while -1:
            _batch = _iter.next()
            x, t = concat_examples(_batch)  # Train Dataset

            with chainer.using_config('train', False):  # dropout を解除する
                y = chainer.cuda.to_cpu(_net(x).data).T[0].tolist()
            list_y = list_y + y
            list_t = list_t + list(t.T[0])

            if _iter.is_new_epoch:
                return numpy.array(list_y), numpy.array(list_t)

    preds_train, factuals_train = pred_value(_net, train_iter)
    preds_test, factuals_test = pred_value(_net, test_iter)

    err_test = numpy.abs(preds_test - factuals_test)  # 0 -- 1 を元に算出
    SEP = numpy.sqrt(numpy.mean(err_test**2))  # 0 -- 1 を元に算出
    r2_test = numpy.corrcoef(preds_test, factuals_test)[0, 1]
    r2_train = numpy.corrcoef(preds_train, factuals_train)[0, 1]

    matplotlib.pyplot.rcParams["font.size"] = 8
    title_contents = "Epochs: %d, SEP: %1.2f [%%]\n\
    R2 score: % 1.3f(train), R2 score: % 1.3f(test)\n\
    (Train, Test) == (%d, %d)"

    fig, ax = matplotlib.pyplot.subplots(1, 1, figsize=(6, 6))
    fig.suptitle(
        title_contents % (train_iter.epoch, SEP, r2_train, r2_test,
                          factuals_train.shape[0], factuals_test.shape[0]))
    try:
        ax.set_xlim(numpy.array([config["xlim_min"], config["xlim_max"]]))
        ax.set_ylim(numpy.array([config["ylim_min"], config["ylim_max"]]))
    except Exception:
        print("err")
        ax.set_xlim(numpy.array([0, 100]))
        ax.set_ylim(numpy.array([0, 100]))
    ax.set_xlabel("Factual Value")
    ax.set_ylabel("Predicted Value")

    # Plotting of result
    # -----------------------
    ax.plot(
        numpy.array(factuals_train),
        numpy.array(preds_train),
        "o",
        label="traing")
    ax.plot(
        numpy.array(factuals_test), numpy.array(preds_test), "x", label="test")
    ax.grid(1)
    ax.legend(loc="upper right")
    ax.set_aspect('equal')
    ax.legend(ncol=2)

    save_img = "%s/PredictPlot.png" % (result_dir)
    fig.savefig(save_img)


class updater_multitask(training.StandardUpdater):
    """ マルチタスク学習用のupdater

    """
    import six
    import numpy
    import chainer
    import chainer.functions as F
    from chainer import cuda, reporter, iterators, serializers
    from chainer.dataset import iterator as iterator_module
    from chainer.dataset import convert, concat_examples

    def __init__(self,
                 iterator,
                 base_cnn,
                 classifiers_model,
                 classifiers_net,
                 base_cnn_optimizer,
                 cl_optimizers,
                 converter=convert.concat_examples,
                 device=-1):
        if isinstance(iterator, iterator_module.Iterator):
            iterator = {'main': iterator}
        self._iterators = iterator
        self.base_cnn = base_cnn
        self.classifiers_model = classifiers_model
        self.classifiers_net = classifiers_net

        self._optimizers = {}
        self._optimizers['base_cnn_opt'] = base_cnn_optimizer
        for i in range(0, len(cl_optimizers)):
            self._optimizers[str(i)] = cl_optimizers[i]

        self.converter = convert.concat_examples
        self.device = device
        self.iteration = 0

    def update_core(self):
        iterator = self._iterators['main'].next()
        in_arrays = self.converter(iterator, self.device)
        xp = numpy if int(self.device) == -1 else cuda.cupy
        x_batch = xp.array(in_arrays[0])
        t_batch = xp.array(in_arrays[1])

        y = self.base_cnn(x_batch)
        loss_dic = {}
        # compute of loss
        for i, classifiers_model in enumerate(self.classifiers_model):
            loss = classifiers_model(y, t_batch[:, i])
            loss_dic[str(i)] = loss
            chainer.reporter.report({'main/loss:%s' % i: loss})

        # compute of accuracy
        for i, classifiers_net in enumerate(self.classifiers_net):
            acc = F.accuracy(classifiers_net(y), t_batch[:, i])
            chainer.reporter.report({'main/acc:%s' % i: acc})

        for name, optimizer in six.iteritems(self._optimizers):
            optimizer.target.cleargrads()

        for name, loss in six.iteritems(loss_dic):
            loss.backward()

        for name, optimizer in six.iteritems(self._optimizers):
            optimizer.update()
