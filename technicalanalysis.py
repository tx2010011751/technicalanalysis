from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# tensorflow BatchNormalization()

import json
import requests
from requests.exceptions import RequestException
import re
import os
import urllib.request
import sys
from urllib.parse import urlencode
import csv
import math
import codecs

import pandas as pd
import numpy as np
np.set_printoptions(threshold=np.inf)
pd.set_option('display.max_rows',None)
pd.set_option('display.max_columns',None)
# 在notebook内绘图
from symfit import parameters, variables, sin, cos, Fit
import matplotlib.pyplot as plt
import time
import datetime
import xlrd
# 设置图形大小
from matplotlib.pylab import rcParams
from tensorflow.contrib import learn

from IPython import embed
import adanet
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import mean_squared_error

from keras.models import Sequential

from keras.layers import Dense, Dropout, LSTM

from scipy.optimize import leastsq

from lstm_returns import LSTM_RETURNS

from tensorflow.python.tools import freeze_graph

rcParams['figure.figsize'] = 40, 20

# 标准化数据
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))

# Data loading params
dev_size = .1

# Parameters
CUT = 1750
PRE = 3
INTEREST = 0.7
TEST = 0
INPUT_N = 150
# ==================================================
# Model Hyperparameters
dropout_prob = 0.5  # 0.5
l2_reg_lambda = 1e-6
learning_rate = 1e-3
#num_hidden = 1008
num_hidden = 1050
n_layers = 2

# Training parameters
batch_size = 1
num_epochs = 1 # 200X
generation = 300
evaluate_every = generation # 100
checkpoint_every = generation# 100
num_checkpoints = generation # Checkpoints to store
TESTREV = 4

B_TRAIN = True

# Misc Parameters
allow_soft_placement = True
log_device_placement = False

saver_path = 'G:\stockmodel'

def get_stock_page(url):
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.132 Safari/537.36'
        }
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            return response.text
        return None
    except RequestException:
        return None

def download_gjcsv(dir):
    url = 'http://quotes.money.163.com/trade/lsjysj_zhishu_000001.html'

    html = get_stock_page(url)

    pattern = re.compile('<input type="radio" name="date_end_type"\s*value="(\d{4}-\d{2}-\d{2})" checked="checked">',
                         re.S)
    end = re.findall(pattern, html)
    end = str(end[0])
    end = end[0:4] + end[5:7] + end[8:10]

    downloadhtml = 'http://quotes.money.163.com/service/chddata.html?' \
                   'code=0000001&start=19901219&end=' + end + '&fields=TCLOSE;HIGH;LOW;TOPEN;LCLOSE;CHG;PCHG;VOTURNOVER;VATURNOVER'

    urllib.request.urlretrieve(downloadhtml, os.path.join(dir + 'gp.csv'))


# Create RSI
def RSI(t, periods=20):
    length = len(t)
    rsies = [np.nan] * length
    # 数据长度不超过周期，无法计算；
    if length <= periods:
        return rsies
    # 用于快速计算；
    up_avg = 0
    down_avg = 0

    # 首先计算第一个RSI，用前periods+1个数据，构成periods个价差序列;
    first_t = t['收盘价'][:periods + 1]
    for i in range(1, len(first_t)):
        # 价格上涨;
        if first_t[i] >= first_t[i - 1]:
            up_avg += first_t[i] - first_t[i - 1]
        # 价格下跌;
        else:
            down_avg += first_t[i - 1] - first_t[i]
    up_avg = up_avg / periods
    down_avg = down_avg / periods
    rs = up_avg / down_avg
    rsies[periods] = 100 - 100 / (1 + rs)

    # 后面的将使用快速计算；
    for j in range(periods + 1, length):
        up = 0
        down = 0
        if t['收盘价'][j] >= t['收盘价'][j - 1]:
            up = t['收盘价'][j] - t['收盘价'][j - 1]
            down = 0
        else:
            up = 0
            down = t['收盘价'][j - 1] - t['收盘价'][j]
        # 类似移动平均的计算公式;
        up_avg = (up_avg * (periods - 1) + up) / periods
        down_avg = (down_avg * (periods - 1) + down) / periods
        rs = up_avg / down_avg
        rsies[j] = 100 - 100 / (1 + rs)
    return rsies


def handleEncoding(original_file):
    # newfile=original_file[0:original_file.rfind(.)]+'_copy.csv'
    f = open(original_file, 'rb+')
    content = f.read()  # 读取文件内容，content为bytes类型，而非string类型
    source_encoding = 'utf-8'
    #####确定encoding类型
    try:
        content.decode('utf-8').encode('utf-8')
        source_encoding = 'utf-8'
    except:
        try:
            content.decode('gbk').encode('utf-8')
            source_encoding = 'gbk'
        except:
            try:
                content.decode('gb2312').encode('utf-8')
                source_encoding = 'gb2312'
            except:
                try:
                    content.decode('gb18030').encode('utf-8')
                    source_encoding = 'gb18030'
                except:
                    try:
                        content.decode('big5').encode('utf-8')
                        source_encoding = 'gb18030'
                    except:
                        content.decode('cp936').encode('utf-8')
                        source_encoding = 'cp936'
    f.close()

    #####按照确定的encoding读取文件内容，并另存为utf-8编码：
    block_size = 4096
    with codecs.open(original_file, 'r', source_encoding) as f:
        with codecs.open(original_file + 'utf8', 'w', 'utf-8') as f2:
            while True:
                content = f.read(block_size)
                if not content:
                    break
                f2.write(content)
    os.remove(original_file)
    os.rename(original_file + 'utf8', original_file)


def sqrfunc(p, x):
    k, b = p
    return k * x + b


def sqrerror(p, x, y):
    return sqrfunc(p, x) - y


def least_sq(x, y):
    p0 = (1, 1)
    Para = leastsq(sqrerror, p0, args=(x, y))
    k, b = Para[0]
    y_pre = k * x[int((PRE - 0.99)/2.0)] + b
    return y_pre

def download_gpcwcsv(cwcode, dir):
    html = 'http://quotes.money.163.com/service/zycwzb_' + cwcode + '.html?type=report'
    urllib.request.urlretrieve(html, os.path.join(dir + cwcode + 'cw.csv'))

    html = 'http://quotes.money.163.com/service/zycwzb_' + cwcode + '.html?type=report&part=ylnl'
    urllib.request.urlretrieve(html, os.path.join(dir + cwcode + 'yl.csv'))

    html = 'http://quotes.money.163.com/service/zycwzb_' + cwcode + '.html?type=report&part=chnl'
    urllib.request.urlretrieve(html, os.path.join(dir + cwcode + 'ch.csv'))

    html = 'http://quotes.money.163.com/service/zycwzb_' + cwcode + '.html?type=report&part=cznl'
    urllib.request.urlretrieve(html, os.path.join(dir + cwcode + 'cz.csv'))

    html = 'http://quotes.money.163.com/service/zycwzb_' + cwcode + '.html?type=report&part=yynl'
    urllib.request.urlretrieve(html, os.path.join(dir + cwcode + 'yy.csv'))

def download_gpgjcsv(gjcode, dir):
    url = 'http://quotes.money.163.com/trade/lsjysj_' + gjcode + '.html#01b07'

    html = get_stock_page(url)

    pattern = re.compile('var STOCKCODE = \'(\d*)\'', re.S)
    code = re.findall(pattern, html)
    code = str(code[0])

    pattern = re.compile('<input type="radio" name="date_start_type" value="(\d{4}-\d{2}-\d{2})" >上市日', re.S)
    start = re.findall(pattern, html)
    start = str(start[0])
    start = start[0:4] + start[5:7] + start[8:10]

    pattern = re.compile('<input type="radio" name="date_end_type" value="(\d{4}-\d{2}-\d{2})">今日', re.S)
    end = re.findall(pattern, html)
    end = str(end[0])
    end = end[0:4] + end[5:7] + end[8:10]


    downloadhtml = 'http://quotes.money.163.com/service/chddata.html?code='+ code + '&start='+ start + '&end='+ end + \
                   '&fields=TCLOSE;HIGH;LOW;TOPEN;LCLOSE;CHG;PCHG;TURNOVER;VOTURNOVER;VATURNOVER;TCAP;MCAP'

    urllib.request.urlretrieve(downloadhtml, os.path.join(dir + gjcode + 'gp.csv'))

def gen_batch(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]



def build_LSTM_model(x_data, y_data, x_pre, b_istrain, gpdir):
    pre_label = []
    label = []
    data = []
    # Randomly shuffle data
    np.random.seed(40)
    shuffle_indices = np.random.permutation(np.arange(len(y_data)))
    x_shuffled = x_data[shuffle_indices]
    y_shuffled = y_data[shuffle_indices]

    # Split train/test set
    # TODO: This is very crude, should use cross-validation
    dev_sample_index = -1 * int(dev_size * float(len(y_data)))

    x_train, x_dev =  x_shuffled[:dev_sample_index],  x_shuffled[dev_sample_index:]
    y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]


    #tmp = tf.batch_gather(realy_dev, tf.reshape(tf.argmax(y_dev, 1),[-1,1]))
    #tmp = tf.reduce_mean(tf.cast(tf.batch_gather(realy_dev, tf.reshape(tf.argmax(y_dev, 1),[-1,1])), dtype=tf.float32))
    #tmp = tf.argmax(y_dev, 1)
    #sess = tf.InteractiveSession()
    #print(tmp.eval())

    #os._exit(0)

    # Training
    # ==================================================
    if b_istrain:
        print("train")
        with tf.Graph().as_default():
            session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
            sess = tf.Session(config=session_conf)
            with sess.as_default():

                #print('x_train.shape[1]',x_train.shape[1], 'y_train.shape[1]',y_train.shape[1], 'x_train.shape[2]',x_train.shape[2],
                #      'l2_reg_lambda',l2_reg_lambda,'num_hidden',num_hidden,'n_layers',n_layers,'y_train',y_train)

                model = LSTM_RETURNS(x_train.shape[1], y_train.shape[1], x_train.shape[2], l2_reg_lambda, num_hidden, n_layers)

                # Define Training procedure
                global_step = tf.Variable(0, name="global_step", trainable=True)

                train_op = tf.train.AdadeltaOptimizer(learning_rate).minimize(model.loss, global_step=global_step)
                # optimizer = tf.train.AdamOptimizer(learning_rate)
                # grads_and_vars = optimizer.compute_gradients(model.loss)
                # train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

                # Output directory for models and summaries
                timestamp = str(int(time.time()))
                out_dir = os.path.abspath(os.path.join(saver_path, "runspostion000001", timestamp))
                print("Writing to {}\n".format(out_dir))

                # Summaries for loss and accuracy
                loss_summary = tf.summary.scalar("loss", model.loss)
                acc_summary = tf.summary.scalar("accuracy", model.accuracy)

                # Train Summaries
                train_summary_op = tf.summary.merge([loss_summary, acc_summary])
                train_summary_dir = os.path.join(out_dir, "summaries", "train")
                train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

                # Dev summaries
                dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
                dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
                dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

                # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
                checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
                checkpoint_prefix = os.path.join(checkpoint_dir, "model")
                if not os.path.exists(checkpoint_dir):
                    os.makedirs(checkpoint_dir)
                saver = tf.train.Saver(tf.global_variables(), max_to_keep=num_checkpoints)

                # TRAINING STEP
                def train_step(x_batch, y_batch, save=False):
                    feed_dict = {
                        model.input_x: x_batch,
                        model.input_y: y_batch,
                        model.dropout_keep_prob: dropout_prob,
                    }
                    # print(y_batch)
                    # system()
                    _, step, summaries, loss, accuracy = sess.run(
                        [train_op, global_step, train_summary_op, model.loss, model.accuracy],
                        feed_dict)
                    #time_str = datetime.datetime.now().isoformat()
                    # print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                    if save:
                        train_summary_writer.add_summary(summaries, step)


                # EVALUATE MODEL
                def dev_step(x_batch, y_batch, writer=None, save=False):
                    feed_dict = {
                        model.input_x: x_batch,
                        model.input_y: y_batch,
                        model.dropout_keep_prob: dropout_prob
                    }
                    # label = []
                    # label.extend(y_batch)
                    _, step, summaries, pre_lab, loss, accuracy = sess.run(
                        [train_op, global_step, train_summary_op, model.predictions, model.loss, model.accuracy], feed_dict)
                    #pre_label = []
                    #pre_label.extend(pre_lab)
                    print('loss', loss, 'accuracy', accuracy,)
                    # print('loss', loss, 'accuracy', accuracy)
                    if save:
                        if writer:
                            writer.add_summary(summaries, step)

                # Initialize all variables
                sess.run(tf.global_variables_initializer())

                # CREATE THE BATCHES GENERATOR
                batches = gen_batch(list(zip(x_train, y_train)), batch_size, num_epochs)
                # TRAIN FOR EACH BATCH
                for batch in batches:
                    x_batch, y_batch = zip(*batch)
                    train_step(x_batch, y_batch)
                    current_step = tf.train.global_step(sess, global_step)
                    if current_step % checkpoint_every == 0:
                        path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                        print("Saved model checkpoint to {}\n".format(path))
                        print("")
                    if current_step % evaluate_every == 0:
                        print("\nEvaluation:")
                        dev_step(x_dev, y_dev, writer=dev_summary_writer)
                        print("")
                        time.sleep(18)



    else:
        print("evoluation")
        with tf.Session() as sess:
            doc_dir = os.path.abspath(os.path.join(saver_path, "runspostion000001"))
            path_list = os.listdir(doc_dir)
            path_list.sort()
            timestamp = path_list[-1]

            out_dir = os.path.abspath(os.path.join(doc_dir, timestamp))
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))

            print(checkpoint_dir)
            save_path = tf.train.latest_checkpoint(checkpoint_dir)
            print(save_path)
            new_saver = tf.train.import_meta_graph(save_path+'.meta')
            new_saver.restore(sess, tf.train.latest_checkpoint(checkpoint_dir))

            graph = tf.get_default_graph()
            #sess.run(tf.global_variables_initializer())

            variables = tf.get_collection(tf.GraphKeys.VARIABLES)
            #for i in variables:
            #    print(i)

            X = graph.get_tensor_by_name('input_x:0')
            keep_prob = graph.get_tensor_by_name('dropout_keep_prob:0')
            #result = graph.get_operation_by_name('max_length').outputs[0]
            print(x_pre, dropout_prob)

            pre_label = sess.run('output/predictions:0', feed_dict={X: x_pre, keep_prob: dropout_prob})
            #pre_label = math.tanh(pre_label)

        print('prediction', pre_label)

        file = open("result.txt", "a+")

        file.write(gpdir + ":  " + str(pre_label)[2:-2] + '\n')

        file.close()
'''
    elif b_continue:
        with tf.Session() as sess:
            doc_dir = os.path.abspath(os.path.join(saver_path, "runs"+ code))
            path_list = os.listdir(doc_dir)
            path_list.sort()
            timestamp = path_list[-1]

            out_dir = os.path.abspath(os.path.join(doc_dir, timestamp))

            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))

            checkpoint_prefix = os.path.join(checkpoint_dir, "model")


            save_path = tf.train.latest_checkpoint(checkpoint_dir)

            new_saver = tf.train.import_meta_graph(save_path+'.meta')
            new_saver.restore(sess, tf.train.latest_checkpoint(checkpoint_dir))

            graph = tf.get_default_graph()
            sess.run(tf.global_variables_initializer())

            #variables = tf.get_collection(tf.GraphKeys.VARIABLES)
            #for i in variables:
            #    print(i)
            input_x = graph.get_tensor_by_name('input_x:0')
            input_y = graph.get_tensor_by_name('input_y:0')
            dropout_keep_prob = graph.get_tensor_by_name('dropout_keep_prob:0')
            predictions = graph.get_tensor_by_name("Variable_3:0")
            model_loss = graph.get_tensor_by_name('loss:0')
            model_accuracy = graph.get_tensor_by_name('accuracy:0')
            model_best_result = graph.get_tensor_by_name('best_result:0')

            global_step = tf.Variable(0, name="global_step", trainable=True)
            system()
            train_op = tf.train.AdadeltaOptimizer(learning_rate).minimize(model_loss, global_step=global_step)

            saver = tf.train.Saver(tf.global_variables(), max_to_keep=num_checkpoints)

            def continue_train_step(x_batch, y_batch, print):
                feed_dict = {
                    input_x: x_batch,
                    input_y: y_batch,
                    dropout_keep_prob: dropout_prob
                }
                # print(y_batch)
                # system()
                if print:
                    loss, accuracy, best_result = sess.run(
                        [model_loss, model_accuracy, model_best_result],
                        feed_dict)
                    print()
                else:
                    sess.run(feed_dict)

            # CREATE THE BATCHES GENERATOR
            batches = gen_batch(list(zip(x_train, y_train)), batch_size, num_epochs)
            # TRAIN FOR EACH BATCH
            for batch in batches:
                continue_train_step(x_batch, y_batch, False)
                current_step = tf.train.global_step(sess, global_step)
                if current_step % evaluate_every == 0:
                    print("\nEvaluation:")
                    continue_train_step(x_dev, y_dev, True)
                    print("")
                    time.sleep(20)
                if current_step % checkpoint_every == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))
            continue_train_step(x_dev, y_dev, True)
'''

def stock_pre(dir):
    ''' #正式使用时开放
    deletline = []

    for i in range(0, len( tmpgpdata)):
        if tmpgpdata['收盘价'][i] == 0:
            deletline.append(i)
            print(i)

    print(len(deletline))
    for i in range(0, len(deletline)):
        tmpgpdata = tmpgpdata.drop(deletline[i])
        tmpdata = tmpdata.drop(deletline[i])
    '''


    # 图表
    #plt.figure(figsize=(40, 20))
    # plt.plot(data['收盘价'])
    # plt.show()
    #single stock processing


    if B_TRAIN:
        file = open("representation.txt", "r")
        selectedgp = []
        for line in file.readlines():
            selectedgp.append(line.strip())

        x_data, y_data = [], []

        for i in range(0, len(selectedgp)):
            # print(selectedgp[i])
            if selectedgp[i] != []:
                selectedgpi = selectedgp[i]
                selectedgpi = selectedgpi.split()

                code = selectedgpi[1]
                gpdir = selectedgpi[2]
                print(code, gpdir)
                #download_gpcwcsv(code, gpdir)
                #download_gpgjcsv(code, gpdir)

                handleEncoding(gpdir + code + 'gp.csv')
                # 读取文件
                gpdata = pd.read_csv(gpdir + code + 'gp.csv', encoding='utf-8')

                gpdata['日期'] = pd.to_datetime(gpdata['日期'], format='%Y/%m/%d')

                gpdata.index = gpdata['日期']

                handleEncoding(dir + 'gp.csv')
                # 读取文件
                data = pd.read_csv(dir + 'gp.csv', encoding='utf-8')

                # 打印头部
                # print(data)
                # 将索引设置为日期
                data['日期'] = \
                    pd.to_datetime(data['日期'], format='%Y/%m/%d')
                # print(data['日期])
                data.index = data['日期']

            deleteline = []
            for i in range(0, len(gpdata)):
                if gpdata['收盘价'][i] == 0:
                    deleteline.append(gpdata['日期'][i])

                if i >= 0 and i <= 10:
                    deleteline.append(gpdata['日期'][i])

            # data = data.drop(deleteline)
            gpdata = gpdata.drop(deleteline)
            data = data.drop(deleteline)

            N0 = len(data) - CUT
            N = len(gpdata) - CUT

            if len(gpdata) > CUT + INPUT_N + 20:
                flag = True

            if flag:

                data = data.sort_index(ascending=True, axis=0)
                # Create 7 and 21 days Moving Average
                data['ma7'] = data['收盘价'].rolling(window=7).mean()
                data['ma21'] = data['收盘价'].rolling(window=21).mean()

                data['ma5'] = data['收盘价'].rolling(window=5).mean()
                data['ma10'] = data['收盘价'].rolling(window=10).mean()
                data['ma20'] = data['收盘价'].rolling(window=20).mean()
                data['ma30'] = data['收盘价'].rolling(window=30).mean()
                data['ma60'] = data['收盘价'].rolling(window=60).mean()
                data['ma120'] = data['收盘价'].rolling(window=120).mean()
                data['ma250'] = data['收盘价'].rolling(window=250).mean()

                data['mad5'] = data['成交量'].rolling(window=5).mean()
                data['mad10'] = data['成交量'].rolling(window=10).mean()
                data['mad20'] = data['成交量'].rolling(window=20).mean()
                data['mad30'] = data['成交量'].rolling(window=30).mean()
                data['mad60'] = data['成交量'].rolling(window=60).mean()

                data['ma5d'] = (data['收盘价']*data['成交量']).rolling(window=5).mean()
                data['ma10d'] = (data['收盘价']*data['成交量']).rolling(window=10).mean()
                data['ma20d'] = (data['收盘价']*data['成交量']).rolling(window=20).mean()
                data['ma30d'] = (data['收盘价']*data['成交量']).rolling(window=30).mean()
                data['ma60d'] = (data['收盘价']*data['成交量']).rolling(window=60).mean()
                data['ma120d'] = (data['收盘价']*data['成交量']).rolling(window=120).mean()
                data['ma250d'] = (data['收盘价']*data['成交量']).rolling(window=250).mean()

                # Create MACD
                data['26ema'] = data['收盘价'].ewm(span=26).mean()
                data['12ema'] = data['收盘价'].ewm(span=12).mean()
                data['MACD'] = data['12ema'] - data['26ema']
                data['DEA'] = pd.Series(data['MACD']).ewm(span=9).mean()

                # Create Bollinger Bands
                data['20sd'] = pd.Series(np.round(data['收盘价'].rolling(20).std(ddof=0), 2))
                data['upper_band'] = data['ma21'] + (data['20sd'] * 2)
                data['lower_band'] = data['ma21'] - (data['20sd'] * 2)

                # Create Exponential moving average
                data['ema'] = data['收盘价'].ewm(com=0.5, adjust=False).mean()

                # Create KDJ
                low_list = data['最低价'].rolling(9, min_periods=9).min()
                low_list.fillna(value=data['最低价'].expanding().min(), inplace=True)
                high_list = data['最高价'].rolling(9, min_periods=9).max()
                high_list.fillna(value=data['最高价'].expanding().max(), inplace=True)
                rsv = (data['收盘价'] - low_list) / (high_list - low_list) * 100

                data['K'] = pd.DataFrame(rsv).ewm(com=2).mean()
                data['D'] = data['K'].ewm(com=2).mean()
                data['J'] = 3 * data['K'] - 2 * data['D']
                data['KDJ'] = 2 * data['J'] - data['K'] - data['D']

                # Create SUPERKDJ
                slow_list = data['最低价'].rolling(60, min_periods=60).min()
                slow_list.fillna(value=data['最低价'].expanding().min(), inplace=True)

                slow_list60 = data['收盘价'].rolling(60, min_periods=60).min()
                slow_list60.fillna(value=data['收盘价'].expanding().min(), inplace=True)
                data['60lowprice'] = (data['收盘价'] - slow_list60) / data['收盘价'] * 100;

                shigh_list = data['最高价'].rolling(60, min_periods=60).max()
                shigh_list.fillna(value=data['最高价'].expanding().max(), inplace=True)

                shigh_list60 = data['收盘价'].rolling(60, min_periods=60).max()
                shigh_list60.fillna(value=data['收盘价'].expanding().max(), inplace=True)
                data['60highprice'] = (shigh_list60 - data['收盘价']) / data['收盘价'] * 100;

                lowamount_list = data['成交量'].rolling(60, min_periods=60).min()
                lowamount_list.fillna(value=data['成交量'].expanding().min(), inplace=True)

                data['60lowamount'] = (data['成交量'] - lowamount_list) / data['成交量'] * 100;

                highamount_list = data['成交量'].rolling(60, min_periods=60).max()
                highamount_list.fillna(value=data['成交量'].expanding().max(), inplace=True)

                data['60highamount'] = (highamount_list - data['成交量']) / data['成交量'] * 100;

                srsv = (data['收盘价'] - slow_list) / (shigh_list - slow_list) * (data['成交量'] - lowamount_list) / (
                    highamount_list - lowamount_list) * 100

                data['SK'] = pd.DataFrame(srsv).ewm(com=5).mean()
                data['SD'] = data['SK'].ewm(com=5).mean()
                data['SJ'] = 3 * data['SK'] - 2 * data['SD']
                data['SKDJ'] = 2 * data['SJ'] - data['SK'] - data['SD']

                # Create LSUPERKDJ
                datelength = 250
                lslow_list = data['最低价'].rolling(datelength, min_periods=datelength).min()
                lslow_list.fillna(value=data['最低价'].expanding().min(), inplace=True)

                slow_list250 = data['收盘价'].rolling(250, min_periods=250).min()
                slow_list250.fillna(value=data['收盘价'].expanding().min(), inplace=True)
                data['250lowprice'] = (data['收盘价'] - slow_list60) / data['收盘价'] * 100;

                lshigh_list = data['最高价'].rolling(datelength, min_periods=datelength).max()
                lshigh_list.fillna(value=data['最高价'].expanding().max(), inplace=True)

                shigh_list250 = data['收盘价'].rolling(250, min_periods=250).max()
                shigh_list250.fillna(value=data['收盘价'].expanding().max(), inplace=True)
                data['250highprice'] = (shigh_list250 - data['收盘价']) / data['收盘价'] * 100;

                llowamount_list = data['成交量'].rolling(datelength, min_periods=datelength).min()
                llowamount_list.fillna(value=data['成交量'].expanding().min(), inplace=True)

                data['250lowamount'] = (data['成交量'] - lowamount_list) / data['成交量'] * 100;

                lhighamount_list = data['成交量'].rolling(datelength, min_periods=datelength).max()
                lhighamount_list.fillna(value=data['成交量'].expanding().max(), inplace=True)

                data['250highamount'] = (highamount_list - data['成交量']) / data['成交量'] * 100;

                lsrsv = ((data['收盘价'] - lslow_list) / (lshigh_list - lslow_list) * (data['成交量'] - llowamount_list) / (
                    lhighamount_list - llowamount_list)) * 100

                data['LSK'] = pd.DataFrame(lsrsv).ewm(com=5).mean()
                data['LSD'] = data['LSK'].ewm(com=5).mean()
                data['LSJ'] = 3 * data['LSK'] - 2 * data['LSD']
                data['LSKDJ'] = 2 * data['LSJ'] - data['LSK'] - data['LSD']

                data['RSI'] = RSI(data)

                # Create OBV
                data['OBV'] = (2.0 * data['收盘价'] - data['最高价'] - data['最低价']) / (data['最高价'] - data['最低价']) * data['成交量']

                # 图表
                # plt.figure(figsize=(40, 20))
                # plt.plot(gpdata['收盘价'])
                # plt.show()

                gpdata = gpdata.sort_index(ascending=True, axis=0)
                # Create 7 and 21 days Moving Average
                gpdata['ma7'] = gpdata['收盘价'].rolling(window=7).mean()
                gpdata['ma21'] = gpdata['收盘价'].rolling(window=21).mean()

                gpdata['ma5'] = gpdata['收盘价'].rolling(window=5).mean()
                gpdata['ma10'] = gpdata['收盘价'].rolling(window=10).mean()
                gpdata['ma20'] = gpdata['收盘价'].rolling(window=20).mean()
                gpdata['ma30'] = gpdata['收盘价'].rolling(window=30).mean()
                gpdata['ma50'] = gpdata['收盘价'].rolling(window=50).mean()
                gpdata['ma60'] = gpdata['收盘价'].rolling(window=60).mean()
                gpdata['ma120'] = gpdata['收盘价'].rolling(window=120).mean()
                gpdata['ma250'] = gpdata['收盘价'].rolling(window=250).mean()

                gpdata['mad5'] = gpdata['成交量'].rolling(window=5).mean()
                gpdata['mad10'] = gpdata['成交量'].rolling(window=10).mean()
                gpdata['mad20'] = gpdata['成交量'].rolling(window=20).mean()
                gpdata['mad30'] = gpdata['成交量'].rolling(window=30).mean()
                gpdata['mad60'] = gpdata['成交量'].rolling(window=60).mean()

                gpdata['ma5d'] = (gpdata['收盘价'] * gpdata['成交量']).rolling(window=5).mean()
                gpdata['ma10d'] = (gpdata['收盘价'] * gpdata['成交量']).rolling(window=10).mean()
                gpdata['ma20d'] = (gpdata['收盘价'] * gpdata['成交量']).rolling(window=20).mean()
                gpdata['ma30d'] = (gpdata['收盘价'] * gpdata['成交量']).rolling(window=30).mean()
                gpdata['ma60d'] = (gpdata['收盘价'] * gpdata['成交量']).rolling(window=60).mean()
                gpdata['ma120d'] = (gpdata['收盘价'] * gpdata['成交量']).rolling(window=120).mean()
                gpdata['ma250d'] = (gpdata['收盘价'] * gpdata['成交量']).rolling(window=250).mean()

                # Create MACD
                gpdata['26ema'] = gpdata['收盘价'].ewm(span=26).mean()
                gpdata['12ema'] = gpdata['收盘价'].ewm(span=12).mean()
                gpdata['MACD'] = (gpdata['12ema'] - gpdata['26ema'])
                gpdata['DEA'] = pd.Series(gpdata['MACD']).ewm(span=9).mean()

                # Create Bollinger Bands
                gpdata['20sd'] = pd.Series(np.round(gpdata['收盘价'].rolling(20).std(ddof=0), 2))
                gpdata['upper_band'] = gpdata['ma21'] + (gpdata['20sd'] * 2)
                gpdata['lower_band'] = gpdata['ma21'] - (gpdata['20sd'] * 2)

                # Create KDJ
                low_list = gpdata['最低价'].rolling(9, min_periods=9).min()
                low_list.fillna(value=gpdata['最低价'].expanding().min(), inplace=True)
                high_list = gpdata['最高价'].rolling(9, min_periods=9).max()
                high_list.fillna(value=gpdata['最高价'].expanding().max(), inplace=True)
                rsv = (gpdata['收盘价'] - low_list) / (high_list - low_list) * 100

                gpdata['K'] = pd.DataFrame(rsv).ewm(com=2).mean()
                gpdata['D'] = gpdata['K'].ewm(com=2).mean()
                gpdata['J'] = 3 * gpdata['K'] - 2 * gpdata['D']
                gpdata['KDJ'] = 2 * gpdata['J'] - gpdata['K'] - gpdata['D']

                # Create SUPERKDJ
                gpslow_list = gpdata['最低价'].rolling(60, min_periods=60).min()
                gpslow_list.fillna(value=gpdata['最低价'].expanding().min(), inplace=True)

                gpslow_list60 = gpdata['收盘价'].rolling(60, min_periods=60).min()
                gpslow_list60.fillna(value=gpdata['收盘价'].expanding().min(), inplace=True)
                gpdata['60lowprice'] = (gpdata['收盘价'] - gpslow_list60) / gpdata['收盘价'] * 100;

                gpshigh_list = gpdata['最高价'].rolling(60, min_periods=60).max()
                gpshigh_list.fillna(value=gpdata['最高价'].expanding().max(), inplace=True)

                gpshigh_list60 = gpdata['收盘价'].rolling(60, min_periods=60).max()
                gpshigh_list60.fillna(value=gpdata['收盘价'].expanding().max(), inplace=True)
                gpdata['60highprice'] = (gpshigh_list60 - gpdata['收盘价']) / gpdata['收盘价'] * 100;

                gplowamount_list = gpdata['成交量'].rolling(60, min_periods=60).min()
                gplowamount_list.fillna(value=gpdata['成交量'].expanding().min(), inplace=True)

                gpdata['60lowamount'] = (gpdata['成交量'] - gplowamount_list) / gpdata['成交量'] * 100;

                gphighamount_list = gpdata['成交量'].rolling(60, min_periods=60).max()
                gphighamount_list.fillna(value=gpdata['成交量'].expanding().max(), inplace=True)

                gpdata['60highamount'] = (gphighamount_list - gpdata['成交量']) / gpdata['成交量'] * 100;

                srsv = (gpdata['收盘价'] - gpslow_list) / (gpshigh_list - gpslow_list) * (gpdata['成交量'] - gplowamount_list) / (
                        gphighamount_list - gplowamount_list) * 100

                gpdata['SK'] = pd.DataFrame(srsv).ewm(com=5).mean()
                gpdata['SD'] = gpdata['SK'].ewm(com=5).mean()
                gpdata['SJ'] = 3 * gpdata['SK'] - 2 * gpdata['SD']
                gpdata['SKDJ'] = 2 * gpdata['SJ'] - gpdata['SK'] - gpdata['SD']

                # Create LSUPERKDJ
                datelength = 250
                gplslow_list = gpdata['最低价'].rolling(datelength, min_periods=datelength).min()
                gplslow_list.fillna(value=gpdata['最低价'].expanding().min(), inplace=True)

                gpslow_list250 = gpdata['收盘价'].rolling(250, min_periods=250).min()
                gpslow_list250.fillna(value=gpdata['收盘价'].expanding().min(), inplace=True)
                gpdata['250lowprice'] = (gpdata['收盘价'] - gpslow_list60) / gpdata['收盘价'] * 100;

                gplshigh_list = gpdata['最高价'].rolling(datelength, min_periods=datelength).max()
                gplshigh_list.fillna(value=gpdata['最高价'].expanding().max(), inplace=True)

                gpshigh_list250 = gpdata['收盘价'].rolling(250, min_periods=250).max()
                gpshigh_list250.fillna(value=gpdata['收盘价'].expanding().max(), inplace=True)
                gpdata['250highprice'] = (gpshigh_list250 - gpdata['收盘价']) / gpdata['收盘价'] * 100;

                gpllowamount_list = gpdata['成交量'].rolling(datelength, min_periods=datelength).min()
                gpllowamount_list.fillna(value=gpdata['成交量'].expanding().min(), inplace=True)

                gpdata['250lowamount'] = (gpdata['成交量'] - gpllowamount_list) / gpdata['成交量'] * 100;

                gplhighamount_list = gpdata['成交量'].rolling(datelength, min_periods=datelength).max()
                gplhighamount_list.fillna(value=gpdata['成交量'].expanding().max(), inplace=True)

                gpdata['250highamount'] = (gphighamount_list - gpdata['成交量']) / gpdata['成交量'] * 100;

                lsrsv = ((gpdata['收盘价'] - gplslow_list) / (gplshigh_list - gplslow_list) *
                     (gpdata['成交量'] - gpllowamount_list) / (
                             gplhighamount_list - gpllowamount_list)) * 100

                gpdata['LSK'] = pd.DataFrame(lsrsv).ewm(com=5).mean()
                gpdata['LSD'] = gpdata['LSK'].ewm(com=5).mean()
                gpdata['LSJ'] = 3 * gpdata['LSK'] - 2 * gpdata['LSD']
                gpdata['LSKDJ'] = 2 * gpdata['LSJ'] - gpdata['LSK'] - gpdata['LSD']

                gpdata['RSI'] = RSI(gpdata)

                # Create OBV
                gpdata['OBV'] = (2.0 * gpdata['收盘价'] - gpdata['最高价'] - gpdata['最低价']) / (gpdata['最高价'] - gpdata['最低价']) * \
                            gpdata['成交量']

                new_gpdata = pd.DataFrame(index=range(0, len(gpdata) - N),
                                      columns=['gp date', 'price', 'high price','low price','5ma', '10ma', '20ma', '30ma', '60ma', '120ma',
                                               '250ma',
                                               'upper_band', 'lower_band', 'amount', '5mad', '10mad', '20mad', '30mad',
                                               '60mad', 'MACD', 'DEA', 'SKDJ', 'LSKDJ', 'RSI',
                                               'pricemad','5maxd', '10maxd', '20maxd', '30maxd', '60maxd', '120maxd','250maxd',

                                               'gp price', 'gp 5ma', 'gp 10ma', 'gp 20ma', 'gp 30ma', 'gp 60ma',
                                               'gp 120ma', 'gp 250ma',
                                               'gp upper_band', 'gp lower_band', 'gp amount', 'gp 5mad', 'gp 10mad',
                                               'gp 20mad', 'gp 30mad','gp 60mad',
                                               'gp pricemad', 'gp 5maxd', 'gp 10maxd', 'gp 20maxd', 'gp 30maxd', 'gp 60maxd',
                                               'gp 120maxd', 'gp 250maxd',
                                               'gp MACD', 'gp DEA', 'gp SKDJ', 'gp LSKDJ','gp RSI', 'gp yh', 'gp yl'])
                '''
                new_gpdata = pd.DataFrame(index=range(0, len(gpdata) - N),
                                    columns=['gp date',
                                            'gp price', 'gp 5ma', 'gp 10ma', 'gp 20ma', 'gp 30ma', 'gp 60ma',
                                            'gp 120ma', 'gp 250ma',
                                            'gp upper_band', 'gp lower_band', 'gp amount', 'gp 5mad', 'gp 10mad',
                                            'gp 20mad', 'gp 30mad',
                                            'gp 60mad', 'gp MACD', 'gp DEA', 'gp KDJ', 'gp SKDJ', 'gp LSKDJ',
                                            'gp RSI',
                                            'gp 60highprice', 'gp 60lowprice', 'gp 60highamount', 'gp 60lowamount',
                                            'gp 250highprice', 'gp 250lowprice', 'gp 250highamount',
                                            'gp 250lowamount'])
                '''


                for i in range(0, len(gpdata) - N - PRE):
                    new_gpdata['gp date'][i] = gpdata['日期'][i + N]
                    #print('gp date',new_gpdata['gp date'][i], data['日期'][i + N0])


                    new_gpdata['price'][i] = (data['收盘价'][i + N0] - data['收盘价'][i + N0 - 1]) / data['收盘价'][i + N0 - 1]
                    new_gpdata['high price'][i] = (data['最高价'][i + N0] - data['收盘价'][i + N0 - 1]) / data['收盘价'][i + N0 - 1]
                    new_gpdata['low price'][i] = (data['最低价'][i + N0] - data['收盘价'][i + N0 - 1]) / data['收盘价'][i + N0 - 1]
                    new_gpdata['5ma'][i] = (data['收盘价'][i + N0] - data['ma5'][i + N0]) / data['收盘价'][i + N0]
                    new_gpdata['10ma'][i] = (data['收盘价'][i + N0] - data['ma10'][i + N0]) / data['收盘价'][i + N0]
                    new_gpdata['20ma'][i] = (data['收盘价'][i + N0] - data['ma20'][i + N0]) / data['收盘价'][i + N0]
                    new_gpdata['30ma'][i] = (data['收盘价'][i + N0] - data['ma30'][i + N0]) / data['收盘价'][i + N0]
                    new_gpdata['60ma'][i] = (data['收盘价'][i + N0] - data['ma60'][i + N0]) / data['收盘价'][i + N0]
                    new_gpdata['120ma'][i] = (data['收盘价'][i + N0] - data['ma120'][i + N0]) / data['收盘价'][i + N0]
                    new_gpdata['250ma'][i] = (data['收盘价'][i + N0] - data['ma250'][i + N0]) / data['收盘价'][i + N0]
                    # print('end price',new_gpdata['end price'][i])
                    new_gpdata['upper_band'][i] = (data['upper_band'][i + N0] - data['收盘价'][i + N0]) / data['收盘价'][i + N0]
                    # print('high price',new_gpdata['high price'][i])
                    new_gpdata['lower_band'][i] = (data['收盘价'][i + N0] - data['lower_band'][i + N0]) / data['收盘价'][i + N0]
                    new_gpdata['amount'][i] = (float(data['成交量'][i + N0]) - float(data['成交量'][i + N0 - 1])) \
                                          / float(data['成交量'][i + N0 - 1])
                    new_gpdata['5mad'][i] = (float(data['成交量'][i + N0]) - data['ma5'][i + N0]) / float(data['成交量'][i + N0])
                    new_gpdata['10mad'][i] = (data['成交量'][i + N0] - data['ma10'][i + N0]) / data['成交量'][i + N0]
                    new_gpdata['20mad'][i] = (data['成交量'][i + N0] - data['ma20'][i + N0]) / data['成交量'][i + N0]
                    new_gpdata['30mad'][i] = (data['成交量'][i + N0] - data['ma30'][i + N0]) / data['成交量'][i + N0]
                    new_gpdata['60mad'][i] = (data['成交量'][i + N0] - data['ma60'][i + N0]) / data['成交量'][i + N0]
                    new_gpdata['pricemad'][i] = (data['收盘价'][i + N0] * data['成交量'][i + N0] - data['收盘价'][i + N0 - 1] * data['成交量'][i + N0 - 1]) / data['收盘价'][i + N0 - 1] / data['成交量'][i + N0 - 1]
                    new_gpdata['5maxd'][i] = (data['收盘价'][i + N0]*data['成交量'][i + N0] - data['ma5d'][i + N0]) / data['收盘价'][i + N0] / data['成交量'][i + N0]
                    new_gpdata['10maxd'][i] = (data['收盘价'][i + N0]*data['成交量'][i + N0] - data['ma10d'][i + N0]) / data['收盘价'][i + N0] / data['成交量'][i + N0]
                    new_gpdata['20maxd'][i] = (data['收盘价'][i + N0]*data['成交量'][i + N0] - data['ma20d'][i + N0]) / data['收盘价'][i + N0] / data['成交量'][i + N0]
                    new_gpdata['30maxd'][i] = (data['收盘价'][i + N0]*data['成交量'][i + N0] - data['ma30d'][i + N0]) / data['收盘价'][i + N0] / data['成交量'][i + N0]
                    new_gpdata['60maxd'][i] = (data['收盘价'][i + N0]*data['成交量'][i + N0] - data['ma60d'][i + N0]) / data['收盘价'][i + N0] / data['成交量'][i + N0]
                    new_gpdata['120maxd'][i] = (data['收盘价'][i + N0]*data['成交量'][i + N0] - data['ma120d'][i + N0]) / data['收盘价'][i + N0] / data['成交量'][i + N0]
                    new_gpdata['250maxd'][i] = (data['收盘价'][i + N0]*data['成交量'][i + N0] - data['ma250d'][i + N0]) / data['收盘价'][i + N0] / data['成交量'][i + N0]
                    new_gpdata['MACD'][i] = data['MACD'][i + N0]
                    new_gpdata['DEA'][i] = data['DEA'][i + N0]
                    new_gpdata['SKDJ'][i] = data['SKDJ'][i + N0]
                    new_gpdata['LSKDJ'][i] = data['LSKDJ'][i + N0]
                    new_gpdata['RSI'][i] = data['RSI'][i + N0]


                for i in range(0, len(gpdata) - N - PRE):
                    new_gpdata['gp price'][i] = (gpdata['收盘价'][i + N] - gpdata['收盘价'][i + N - 1]) / gpdata['收盘价'][i + N - 1]
                    new_gpdata['gp 5ma'][i] = (gpdata['收盘价'][i + N] - gpdata['ma5'][i + N]) / gpdata['收盘价'][i + N]
                    new_gpdata['gp 10ma'][i] = (gpdata['收盘价'][i + N] - gpdata['ma10'][i + N]) / gpdata['收盘价'][i + N]
                    new_gpdata['gp 20ma'][i] = (gpdata['收盘价'][i + N] - gpdata['ma20'][i + N]) / gpdata['收盘价'][i + N]
                    new_gpdata['gp 30ma'][i] = (gpdata['收盘价'][i + N] - gpdata['ma30'][i + N]) / gpdata['收盘价'][i + N]
                    new_gpdata['gp 60ma'][i] = (gpdata['收盘价'][i + N] - gpdata['ma60'][i + N]) / gpdata['收盘价'][i + N]
                    new_gpdata['gp 120ma'][i] = (gpdata['收盘价'][i + N] - gpdata['ma120'][i + N]) / gpdata['收盘价'][i + N]
                    new_gpdata['gp 250ma'][i] = (gpdata['收盘价'][i + N] - gpdata['ma250'][i + N]) / gpdata['收盘价'][i + N]
                    # print('end gp price',new_gpdata['end gp price'][i])
                    new_gpdata['gp upper_band'][i] = (gpdata['upper_band'][i + N] - gpdata['收盘价'][i + N]) / \
                                                 gpdata['收盘价'][i + N]
                    # print('high gp price',new_gpdata['high gp price'][i])
                    new_gpdata['gp lower_band'][i] = (gpdata['收盘价'][i + N] - gpdata['lower_band'][i + N]) / \
                                                 gpdata['收盘价'][i + N]
                    # print(i + N - 1, gpdata['成交量'][i + N - 1])
                    new_gpdata['gp amount'][i] = (float(gpdata['成交量'][i + N]) - float(
                        gpdata['成交量'][i + N - 1])) / float(
                        gpdata['成交量'][i + N - 1])
                    new_gpdata['gp 5mad'][i] = (gpdata['成交量'][i + N] - gpdata['mad5'][i + N]) / gpdata['成交量'][i + N]
                    new_gpdata['gp 10mad'][i] = (gpdata['成交量'][i + N] - gpdata['mad10'][i + N]) / gpdata['成交量'][i + N]
                    new_gpdata['gp 20mad'][i] = (gpdata['成交量'][i + N] - gpdata['mad20'][i + N]) / gpdata['成交量'][i + N]
                    new_gpdata['gp 30mad'][i] = (gpdata['成交量'][i + N] - gpdata['mad30'][i + N]) / gpdata['成交量'][i + N]
                    new_gpdata['gp 60mad'][i] = (gpdata['成交量'][i + N] - gpdata['mad60'][i + N]) / gpdata['成交量'][i + N]

                    new_gpdata['gp pricemad'][i] = (gpdata['收盘价'][i + N] * gpdata['成交量'][i + N] - gpdata['收盘价'][i + N - 1] *
                                                    gpdata['成交量'][i + N - 1]) / gpdata['收盘价'][i + N - 1] / gpdata['成交量'][i + N - 1]
                    new_gpdata['gp 5maxd'][i] = (gpdata['收盘价'][i + N] * gpdata['成交量'][i + N] - gpdata['ma5d'][i + N]) / \
                                             gpdata['收盘价'][i + N] / gpdata['成交量'][i + N]
                    new_gpdata['gp 10maxd'][i] = (gpdata['收盘价'][i + N] * gpdata['成交量'][i + N] - gpdata['ma10d'][i + N]) / \
                                              gpdata['收盘价'][i + N] / gpdata['成交量'][i + N]
                    new_gpdata['gp 20maxd'][i] = (gpdata['收盘价'][i + N] * gpdata['成交量'][i + N] - gpdata['ma20d'][i + N]) / \
                                              gpdata['收盘价'][i + N] / gpdata['成交量'][i + N]
                    new_gpdata['gp 30maxd'][i] = (gpdata['收盘价'][i + N] * gpdata['成交量'][i + N] - gpdata['ma30d'][i + N]) / \
                                              gpdata['收盘价'][i + N] / gpdata['成交量'][i + N]
                    new_gpdata['gp 60maxd'][i] = (gpdata['收盘价'][i + N] * gpdata['成交量'][i + N] - gpdata['ma60d'][i + N]) / \
                                              gpdata['收盘价'][i + N] / gpdata['成交量'][i + N]
                    new_gpdata['gp 120maxd'][i] = (gpdata['收盘价'][i + N] * gpdata['成交量'][i + N] - gpdata['ma120d'][i + N]) / \
                                               gpdata['收盘价'][i + N] / gpdata['成交量'][i + N]
                    new_gpdata['gp 250maxd'][i] = (gpdata['收盘价'][i + N] * gpdata['成交量'][i + N] - gpdata['ma250d'][i + N]) / \
                                               gpdata['收盘价'][i + N] / gpdata['成交量'][i + N]

                    new_gpdata['gp MACD'][i] = gpdata['MACD'][i + N]
                    new_gpdata['gp DEA'][i] = gpdata['DEA'][i + N]
                    new_gpdata['gp SKDJ'][i] = gpdata['SKDJ'][i + N]
                    new_gpdata['gp LSKDJ'][i] = gpdata['LSKDJ'][i + N]
                    new_gpdata['gp RSI'][i] = gpdata['RSI'][i + N]

                #label_data = pd.DataFrame(index=range(0, len(gpdata) - N - PRE - 1), columns=['growth rate'])
                tmp_y = []; tmp_yave = []; highpriceave =0.0; lowpriceave =0.0

                for i in range(0, len(gpdata) - N):
                    #print(gpdata['日期'][i + N],(max(gpdata['最高价'][i + N + 1 - PRE:i + N + 1]) - gpdata['收盘价'][i + N - PRE])/ gpdata['收盘价'][i + N - PRE]*100,min(gpdata['最低价'][i + N + 1 - PRE:i + N + 1]) ,gpdata['收盘价'][i + N - PRE])
                    tmp_y.append(
                        [ (max(gpdata['最高价'][i + N + 1 - PRE:i + N + 1]) - gpdata['收盘价'][i + N - PRE])/ gpdata['收盘价'][i + N - PRE]*100
                        ,(min(gpdata['最低价'][i + N + 1 - PRE:i + N + 1]) - gpdata['收盘价'][i + N - PRE])/ gpdata['收盘价'][i + N - PRE]*100 ] )

                for i in range(0, len(gpdata) - N - PRE):
                    new_gpdata['gp yh'][i] = tmp_y[i + PRE][0]
                    new_gpdata['gp yl'][i] = tmp_y[i + PRE][1]

                new_gpdata.drop('gp date', axis=1, inplace=True)

                #new_gpdata.isna().any())

                dataset = new_gpdata.values

                train = dataset[:, :]
                # valid = dataset[1900:, :]
                scalerx = MinMaxScaler(feature_range=(0, 1))

                scaled_data = scalerx.fit_transform(dataset)

                # x个参数 INPUT_N个数据组作为预估
                for i in range(INPUT_N , len(train) - PRE):
                    x_data.append(scaled_data[i - INPUT_N: i])
                    # print('x_train',x_train[i - 60])
                    y_data.append(tmp_y[i + PRE])

        x_data, y_data = np.array(x_data), np.array(y_data)

        x_data = np.reshape(x_data, (x_data.shape[0], x_data.shape[1], x_data.shape[2]))
        #print(x_data, x_data.shape[0], x_data.shape[1], x_data.shape[2])

        y_data = np.reshape(y_data, (y_data.shape[0], y_data.shape[1]))
        #print(y_data, y_data.shape[0], y_data.shape[1])

        x_pre = []
        x_pre.append(scaled_data[len(train) - INPUT_N: len(train)])
        #print(x_pre)
        # print('reachbegin')

        build_LSTM_model(x_data, y_data, x_pre, B_TRAIN, code)
        # build_LSTM_model(x_data, y_data, x_pre, False, False, code)

    else:
        file = open("representation.txt", "r")
        selectedgp = []
        for line in file.readlines():
            selectedgp.append(line.strip())

        for i in range(0, len(selectedgp)):
            x_data, y_data = [], []
            # print(selectedgp[i])
            if selectedgp[i] != []:
                selectedgpi = selectedgp[i]
                selectedgpi = selectedgpi.split()

                code = selectedgpi[1]
                gpdir = selectedgpi[2]
                print(code, gpdir)
                # download_gpcwcsv(code, gpdir)
                # download_gpgjcsv(code, gpdir)

                handleEncoding(gpdir + code + 'gp.csv')
                # 读取文件
                gpdata = pd.read_csv(gpdir + code + 'gp.csv', encoding='utf-8')

                gpdata['日期'] = pd.to_datetime(gpdata['日期'], format='%Y/%m/%d')

                gpdata.index = gpdata['日期']

                handleEncoding(dir + 'gp.csv')
                # 读取文件
                data = pd.read_csv(dir + 'gp.csv', encoding='utf-8')

                # 打印头部
                # print(data)
                # 将索引设置为日期
                data['日期'] = \
                    pd.to_datetime(data['日期'], format='%Y/%m/%d')
                # print(data['日期])
                data.index = data['日期']

                deleteline = []
                for i in range(0, len(gpdata)):
                    if gpdata['收盘价'][i] == 0:
                        deleteline.append(gpdata['日期'][i])

                # data = data.drop(deleteline)
                gpdata = gpdata.drop(deleteline)
                data = data.drop(deleteline)

                N0 = len(data) - CUT
                N = len(gpdata) - CUT

                if len(gpdata) > CUT + INPUT_N + 20:
                    flag = True

                if flag:

                    data = data.sort_index(ascending=True, axis=0)
                    # Create 7 and 21 days Moving Average
                    data['ma7'] = data['收盘价'].rolling(window=7).mean()
                    data['ma21'] = data['收盘价'].rolling(window=21).mean()

                    data['ma5'] = data['收盘价'].rolling(window=5).mean()
                    data['ma10'] = data['收盘价'].rolling(window=10).mean()
                    data['ma20'] = data['收盘价'].rolling(window=20).mean()
                    data['ma30'] = data['收盘价'].rolling(window=30).mean()
                    data['ma60'] = data['收盘价'].rolling(window=60).mean()
                    data['ma120'] = data['收盘价'].rolling(window=120).mean()
                    data['ma250'] = data['收盘价'].rolling(window=250).mean()

                    data['mad5'] = data['成交量'].rolling(window=5).mean()
                    data['mad10'] = data['成交量'].rolling(window=10).mean()
                    data['mad20'] = data['成交量'].rolling(window=20).mean()
                    data['mad30'] = data['成交量'].rolling(window=30).mean()
                    data['mad60'] = data['成交量'].rolling(window=60).mean()

                    data['ma5d'] = (data['收盘价'] * data['成交量']).rolling(window=5).mean()
                    data['ma10d'] = (data['收盘价'] * data['成交量']).rolling(window=10).mean()
                    data['ma20d'] = (data['收盘价'] * data['成交量']).rolling(window=20).mean()
                    data['ma30d'] = (data['收盘价'] * data['成交量']).rolling(window=30).mean()
                    data['ma60d'] = (data['收盘价'] * data['成交量']).rolling(window=60).mean()
                    data['ma120d'] = (data['收盘价'] * data['成交量']).rolling(window=120).mean()
                    data['ma250d'] = (data['收盘价'] * data['成交量']).rolling(window=250).mean()

                    # Create MACD
                    data['26ema'] = data['收盘价'].ewm(span=26).mean()
                    data['12ema'] = data['收盘价'].ewm(span=12).mean()
                    data['MACD'] = data['12ema'] - data['26ema']
                    data['DEA'] = pd.Series(data['MACD']).ewm(span=9).mean()

                    # Create Bollinger Bands
                    data['20sd'] = pd.Series(np.round(data['收盘价'].rolling(20).std(ddof=0), 2))
                    data['upper_band'] = data['ma21'] + (data['20sd'] * 2)
                    data['lower_band'] = data['ma21'] - (data['20sd'] * 2)

                    # Create Exponential moving average
                    data['ema'] = data['收盘价'].ewm(com=0.5, adjust=False).mean()

                    # Create KDJ
                    low_list = data['最低价'].rolling(9, min_periods=9).min()
                    low_list.fillna(value=data['最低价'].expanding().min(), inplace=True)
                    high_list = data['最高价'].rolling(9, min_periods=9).max()
                    high_list.fillna(value=data['最高价'].expanding().max(), inplace=True)
                    rsv = (data['收盘价'] - low_list) / (high_list - low_list) * 100

                    data['K'] = pd.DataFrame(rsv).ewm(com=2).mean()
                    data['D'] = data['K'].ewm(com=2).mean()
                    data['J'] = 3 * data['K'] - 2 * data['D']
                    data['KDJ'] = 2 * data['J'] - data['K'] - data['D']

                    # Create SUPERKDJ
                    slow_list = data['最低价'].rolling(60, min_periods=60).min()
                    slow_list.fillna(value=data['最低价'].expanding().min(), inplace=True)

                    slow_list60 = data['收盘价'].rolling(60, min_periods=60).min()
                    slow_list60.fillna(value=data['收盘价'].expanding().min(), inplace=True)
                    data['60lowprice'] = (data['收盘价'] - slow_list60) / data['收盘价'] * 100;

                    shigh_list = data['最高价'].rolling(60, min_periods=60).max()
                    shigh_list.fillna(value=data['最高价'].expanding().max(), inplace=True)

                    shigh_list60 = data['收盘价'].rolling(60, min_periods=60).max()
                    shigh_list60.fillna(value=data['收盘价'].expanding().max(), inplace=True)
                    data['60highprice'] = (shigh_list60 - data['收盘价']) / data['收盘价'] * 100;

                    lowamount_list = data['成交量'].rolling(60, min_periods=60).min()
                    lowamount_list.fillna(value=data['成交量'].expanding().min(), inplace=True)

                    data['60lowamount'] = (data['成交量'] - lowamount_list) / data['成交量'] * 100;

                    highamount_list = data['成交量'].rolling(60, min_periods=60).max()
                    highamount_list.fillna(value=data['成交量'].expanding().max(), inplace=True)

                    data['60highamount'] = (highamount_list - data['成交量']) / data['成交量'] * 100;

                    srsv = (data['收盘价'] - slow_list) / (shigh_list - slow_list) * (data['成交量'] - lowamount_list) / (
                            highamount_list - lowamount_list) * 100

                    data['SK'] = pd.DataFrame(srsv).ewm(com=5).mean()
                    data['SD'] = data['SK'].ewm(com=5).mean()
                    data['SJ'] = 3 * data['SK'] - 2 * data['SD']
                    data['SKDJ'] = 2 * data['SJ'] - data['SK'] - data['SD']

                    # Create LSUPERKDJ
                    datelength = 250
                    lslow_list = data['最低价'].rolling(datelength, min_periods=datelength).min()
                    lslow_list.fillna(value=data['最低价'].expanding().min(), inplace=True)

                    slow_list250 = data['收盘价'].rolling(250, min_periods=250).min()
                    slow_list250.fillna(value=data['收盘价'].expanding().min(), inplace=True)
                    data['250lowprice'] = (data['收盘价'] - slow_list60) / data['收盘价'] * 100;

                    lshigh_list = data['最高价'].rolling(datelength, min_periods=datelength).max()
                    lshigh_list.fillna(value=data['最高价'].expanding().max(), inplace=True)

                    shigh_list250 = data['收盘价'].rolling(250, min_periods=250).max()
                    shigh_list250.fillna(value=data['收盘价'].expanding().max(), inplace=True)
                    data['250highprice'] = (shigh_list250 - data['收盘价']) / data['收盘价'] * 100;

                    llowamount_list = data['成交量'].rolling(datelength, min_periods=datelength).min()
                    llowamount_list.fillna(value=data['成交量'].expanding().min(), inplace=True)

                    data['250lowamount'] = (data['成交量'] - lowamount_list) / data['成交量'] * 100;

                    lhighamount_list = data['成交量'].rolling(datelength, min_periods=datelength).max()
                    lhighamount_list.fillna(value=data['成交量'].expanding().max(), inplace=True)

                    data['250highamount'] = (highamount_list - data['成交量']) / data['成交量'] * 100;

                    lsrsv = ((data['收盘价'] - lslow_list) / (lshigh_list - lslow_list) * (
                                data['成交量'] - llowamount_list) / (
                                     lhighamount_list - llowamount_list)) * 100

                    data['LSK'] = pd.DataFrame(lsrsv).ewm(com=5).mean()
                    data['LSD'] = data['LSK'].ewm(com=5).mean()
                    data['LSJ'] = 3 * data['LSK'] - 2 * data['LSD']
                    data['LSKDJ'] = 2 * data['LSJ'] - data['LSK'] - data['LSD']

                    data['RSI'] = RSI(data)

                    # Create OBV
                    data['OBV'] = (2.0 * data['收盘价'] - data['最高价'] - data['最低价']) / (data['最高价'] - data['最低价']) * data[
                        '成交量']

                    # 图表
                    # plt.figure(figsize=(40, 20))
                    # plt.plot(gpdata['收盘价'])
                    # plt.show()

                    gpdata = gpdata.sort_index(ascending=True, axis=0)
                    # Create 7 and 21 days Moving Average
                    gpdata['ma7'] = gpdata['收盘价'].rolling(window=7).mean()
                    gpdata['ma21'] = gpdata['收盘价'].rolling(window=21).mean()

                    gpdata['ma5'] = gpdata['收盘价'].rolling(window=5).mean()
                    gpdata['ma10'] = gpdata['收盘价'].rolling(window=10).mean()
                    gpdata['ma20'] = gpdata['收盘价'].rolling(window=20).mean()
                    gpdata['ma30'] = gpdata['收盘价'].rolling(window=30).mean()
                    gpdata['ma50'] = gpdata['收盘价'].rolling(window=50).mean()
                    gpdata['ma60'] = gpdata['收盘价'].rolling(window=60).mean()
                    gpdata['ma120'] = gpdata['收盘价'].rolling(window=120).mean()
                    gpdata['ma250'] = gpdata['收盘价'].rolling(window=250).mean()

                    gpdata['mad5'] = gpdata['成交量'].rolling(window=5).mean()
                    gpdata['mad10'] = gpdata['成交量'].rolling(window=10).mean()
                    gpdata['mad20'] = gpdata['成交量'].rolling(window=20).mean()
                    gpdata['mad30'] = gpdata['成交量'].rolling(window=30).mean()
                    gpdata['mad60'] = gpdata['成交量'].rolling(window=60).mean()

                    gpdata['ma5d'] = (gpdata['收盘价'] * gpdata['成交量']).rolling(window=5).mean()
                    gpdata['ma10d'] = (gpdata['收盘价'] * gpdata['成交量']).rolling(window=10).mean()
                    gpdata['ma20d'] = (gpdata['收盘价'] * gpdata['成交量']).rolling(window=20).mean()
                    gpdata['ma30d'] = (gpdata['收盘价'] * gpdata['成交量']).rolling(window=30).mean()
                    gpdata['ma60d'] = (gpdata['收盘价'] * gpdata['成交量']).rolling(window=60).mean()
                    gpdata['ma120d'] = (gpdata['收盘价'] * gpdata['成交量']).rolling(window=120).mean()
                    gpdata['ma250d'] = (gpdata['收盘价'] * gpdata['成交量']).rolling(window=250).mean()

                    # Create MACD
                    gpdata['26ema'] = gpdata['收盘价'].ewm(span=26).mean()
                    gpdata['12ema'] = gpdata['收盘价'].ewm(span=12).mean()
                    gpdata['MACD'] = (gpdata['12ema'] - gpdata['26ema'])
                    gpdata['DEA'] = pd.Series(gpdata['MACD']).ewm(span=9).mean()

                    # Create Bollinger Bands
                    gpdata['20sd'] = pd.Series(np.round(gpdata['收盘价'].rolling(20).std(ddof=0), 2))
                    gpdata['upper_band'] = gpdata['ma21'] + (gpdata['20sd'] * 2)
                    gpdata['lower_band'] = gpdata['ma21'] - (gpdata['20sd'] * 2)

                    # Create KDJ
                    low_list = gpdata['最低价'].rolling(9, min_periods=9).min()
                    low_list.fillna(value=gpdata['最低价'].expanding().min(), inplace=True)
                    high_list = gpdata['最高价'].rolling(9, min_periods=9).max()
                    high_list.fillna(value=gpdata['最高价'].expanding().max(), inplace=True)
                    rsv = (gpdata['收盘价'] - low_list) / (high_list - low_list) * 100

                    gpdata['K'] = pd.DataFrame(rsv).ewm(com=2).mean()
                    gpdata['D'] = gpdata['K'].ewm(com=2).mean()
                    gpdata['J'] = 3 * gpdata['K'] - 2 * gpdata['D']
                    gpdata['KDJ'] = 2 * gpdata['J'] - gpdata['K'] - gpdata['D']

                    # Create SUPERKDJ
                    gpslow_list = gpdata['最低价'].rolling(60, min_periods=60).min()
                    gpslow_list.fillna(value=gpdata['最低价'].expanding().min(), inplace=True)

                    gpslow_list60 = gpdata['收盘价'].rolling(60, min_periods=60).min()
                    gpslow_list60.fillna(value=gpdata['收盘价'].expanding().min(), inplace=True)
                    gpdata['60lowprice'] = (gpdata['收盘价'] - gpslow_list60) / gpdata['收盘价'] * 100;

                    gpshigh_list = gpdata['最高价'].rolling(60, min_periods=60).max()
                    gpshigh_list.fillna(value=gpdata['最高价'].expanding().max(), inplace=True)

                    gpshigh_list60 = gpdata['收盘价'].rolling(60, min_periods=60).max()
                    gpshigh_list60.fillna(value=gpdata['收盘价'].expanding().max(), inplace=True)
                    gpdata['60highprice'] = (gpshigh_list60 - gpdata['收盘价']) / gpdata['收盘价'] * 100;

                    gplowamount_list = gpdata['成交量'].rolling(60, min_periods=60).min()
                    gplowamount_list.fillna(value=gpdata['成交量'].expanding().min(), inplace=True)

                    gpdata['60lowamount'] = (gpdata['成交量'] - gplowamount_list) / gpdata['成交量'] * 100;

                    gphighamount_list = gpdata['成交量'].rolling(60, min_periods=60).max()
                    gphighamount_list.fillna(value=gpdata['成交量'].expanding().max(), inplace=True)

                    gpdata['60highamount'] = (gphighamount_list - gpdata['成交量']) / gpdata['成交量'] * 100;

                    srsv = (gpdata['收盘价'] - gpslow_list) / (gpshigh_list - gpslow_list) * (
                                gpdata['成交量'] - gplowamount_list) / (
                                   gphighamount_list - gplowamount_list) * 100

                    gpdata['SK'] = pd.DataFrame(srsv).ewm(com=5).mean()
                    gpdata['SD'] = gpdata['SK'].ewm(com=5).mean()
                    gpdata['SJ'] = 3 * gpdata['SK'] - 2 * gpdata['SD']
                    gpdata['SKDJ'] = 2 * gpdata['SJ'] - gpdata['SK'] - gpdata['SD']

                    # Create LSUPERKDJ
                    datelength = 250
                    gplslow_list = gpdata['最低价'].rolling(datelength, min_periods=datelength).min()
                    gplslow_list.fillna(value=gpdata['最低价'].expanding().min(), inplace=True)

                    gpslow_list250 = gpdata['收盘价'].rolling(250, min_periods=250).min()
                    gpslow_list250.fillna(value=gpdata['收盘价'].expanding().min(), inplace=True)
                    gpdata['250lowprice'] = (gpdata['收盘价'] - gpslow_list60) / gpdata['收盘价'] * 100;

                    gplshigh_list = gpdata['最高价'].rolling(datelength, min_periods=datelength).max()
                    gplshigh_list.fillna(value=gpdata['最高价'].expanding().max(), inplace=True)

                    gpshigh_list250 = gpdata['收盘价'].rolling(250, min_periods=250).max()
                    gpshigh_list250.fillna(value=gpdata['收盘价'].expanding().max(), inplace=True)
                    gpdata['250highprice'] = (gpshigh_list250 - gpdata['收盘价']) / gpdata['收盘价'] * 100;

                    gpllowamount_list = gpdata['成交量'].rolling(datelength, min_periods=datelength).min()
                    gpllowamount_list.fillna(value=gpdata['成交量'].expanding().min(), inplace=True)

                    gpdata['250lowamount'] = (gpdata['成交量'] - gpllowamount_list) / gpdata['成交量'] * 100;

                    gplhighamount_list = gpdata['成交量'].rolling(datelength, min_periods=datelength).max()
                    gplhighamount_list.fillna(value=gpdata['成交量'].expanding().max(), inplace=True)

                    gpdata['250highamount'] = (gphighamount_list - gpdata['成交量']) / gpdata['成交量'] * 100;

                    lsrsv = ((gpdata['收盘价'] - gplslow_list) / (gplshigh_list - gplslow_list) *
                             (gpdata['成交量'] - gpllowamount_list) / (
                                     gplhighamount_list - gpllowamount_list)) * 100

                    gpdata['LSK'] = pd.DataFrame(lsrsv).ewm(com=5).mean()
                    gpdata['LSD'] = gpdata['LSK'].ewm(com=5).mean()
                    gpdata['LSJ'] = 3 * gpdata['LSK'] - 2 * gpdata['LSD']
                    gpdata['LSKDJ'] = 2 * gpdata['LSJ'] - gpdata['LSK'] - gpdata['LSD']

                    gpdata['RSI'] = RSI(gpdata)

                    # Create OBV
                    gpdata['OBV'] = (2.0 * gpdata['收盘价'] - gpdata['最高价'] - gpdata['最低价']) / (
                                gpdata['最高价'] - gpdata['最低价']) * \
                                    gpdata['成交量']

                    new_gpdata = pd.DataFrame(index=range(0, len(gpdata) - N),
                                              columns=['gp date', 'price', 'high price', 'low price', '5ma', '10ma',
                                                       '20ma', '30ma', '60ma', '120ma',
                                                       '250ma',
                                                       'upper_band', 'lower_band', 'amount', '5mad', '10mad', '20mad',
                                                       '30mad',
                                                       '60mad', 'MACD', 'DEA', 'SKDJ', 'LSKDJ', 'RSI',
                                                       'pricemad', '5maxd', '10maxd', '20maxd', '30maxd', '60maxd',
                                                       '120maxd', '250maxd',

                                                       'gp price', 'gp 5ma', 'gp 10ma', 'gp 20ma', 'gp 30ma', 'gp 60ma',
                                                       'gp 120ma', 'gp 250ma',
                                                       'gp upper_band', 'gp lower_band', 'gp amount', 'gp 5mad',
                                                       'gp 10mad',
                                                       'gp 20mad', 'gp 30mad', 'gp 60mad',
                                                       'gp pricemad', 'gp 5maxd', 'gp 10maxd', 'gp 20maxd', 'gp 30maxd',
                                                       'gp 60maxd',
                                                       'gp 120maxd', 'gp 250maxd',
                                                       'gp MACD', 'gp DEA', 'gp SKDJ', 'gp LSKDJ', 'gp RSI'])
                    '''
                    new_gpdata = pd.DataFrame(index=range(0, len(gpdata) - N),
                                        columns=['gp date',
                                                'gp price', 'gp 5ma', 'gp 10ma', 'gp 20ma', 'gp 30ma', 'gp 60ma',
                                                'gp 120ma', 'gp 250ma',
                                                'gp upper_band', 'gp lower_band', 'gp amount', 'gp 5mad', 'gp 10mad',
                                                'gp 20mad', 'gp 30mad',
                                                'gp 60mad', 'gp MACD', 'gp DEA', 'gp KDJ', 'gp SKDJ', 'gp LSKDJ',
                                                'gp RSI',
                                                'gp 60highprice', 'gp 60lowprice', 'gp 60highamount', 'gp 60lowamount',
                                                'gp 250highprice', 'gp 250lowprice', 'gp 250highamount',
                                                'gp 250lowamount'])
                    '''

                    for i in range(0, len(gpdata) - N - PRE):
                        new_gpdata['gp date'][i] = gpdata['日期'][i + N]
                        # print('gp date',new_gpdata['gp date'][i], data['日期'][i + N0])

                        new_gpdata['price'][i] = (data['收盘价'][i + N0] - data['收盘价'][i + N0 - 1]) / data['收盘价'][
                            i + N0 - 1]
                        new_gpdata['high price'][i] = (data['最高价'][i + N0] - data['收盘价'][i + N0 - 1]) / data['收盘价'][
                            i + N0 - 1]
                        new_gpdata['low price'][i] = (data['最低价'][i + N0] - data['收盘价'][i + N0 - 1]) / data['收盘价'][
                            i + N0 - 1]
                        new_gpdata['5ma'][i] = (data['收盘价'][i + N0] - data['ma5'][i + N0]) / data['收盘价'][i + N0]
                        new_gpdata['10ma'][i] = (data['收盘价'][i + N0] - data['ma10'][i + N0]) / data['收盘价'][i + N0]
                        new_gpdata['20ma'][i] = (data['收盘价'][i + N0] - data['ma20'][i + N0]) / data['收盘价'][i + N0]
                        new_gpdata['30ma'][i] = (data['收盘价'][i + N0] - data['ma30'][i + N0]) / data['收盘价'][i + N0]
                        new_gpdata['60ma'][i] = (data['收盘价'][i + N0] - data['ma60'][i + N0]) / data['收盘价'][i + N0]
                        new_gpdata['120ma'][i] = (data['收盘价'][i + N0] - data['ma120'][i + N0]) / data['收盘价'][i + N0]
                        new_gpdata['250ma'][i] = (data['收盘价'][i + N0] - data['ma250'][i + N0]) / data['收盘价'][i + N0]
                        # print('end price',new_gpdata['end price'][i])
                        new_gpdata['upper_band'][i] = (data['upper_band'][i + N0] - data['收盘价'][i + N0]) / data['收盘价'][
                            i + N0]
                        # print('high price',new_gpdata['high price'][i])
                        new_gpdata['lower_band'][i] = (data['收盘价'][i + N0] - data['lower_band'][i + N0]) / data['收盘价'][
                            i + N0]
                        new_gpdata['amount'][i] = (float(data['成交量'][i + N0]) - float(data['成交量'][i + N0 - 1])) \
                                                  / float(data['成交量'][i + N0 - 1])
                        new_gpdata['5mad'][i] = (float(data['成交量'][i + N0]) - data['ma5'][i + N0]) / float(
                            data['成交量'][i + N0])
                        new_gpdata['10mad'][i] = (data['成交量'][i + N0] - data['ma10'][i + N0]) / data['成交量'][i + N0]
                        new_gpdata['20mad'][i] = (data['成交量'][i + N0] - data['ma20'][i + N0]) / data['成交量'][i + N0]
                        new_gpdata['30mad'][i] = (data['成交量'][i + N0] - data['ma30'][i + N0]) / data['成交量'][i + N0]
                        new_gpdata['60mad'][i] = (data['成交量'][i + N0] - data['ma60'][i + N0]) / data['成交量'][i + N0]
                        new_gpdata['pricemad'][i] = (data['收盘价'][i + N0] * data['成交量'][i + N0] - data['收盘价'][
                            i + N0 - 1] * data['成交量'][i + N0 - 1]) / data['收盘价'][i + N0 - 1] / data['成交量'][i + N0 - 1]
                        new_gpdata['5maxd'][i] = (data['收盘价'][i + N0] * data['成交量'][i + N0] - data['ma5d'][i + N0]) / \
                                                 data['收盘价'][i + N0] / data['成交量'][i + N0]
                        new_gpdata['10maxd'][i] = (data['收盘价'][i + N0] * data['成交量'][i + N0] - data['ma10d'][i + N0]) / \
                                                  data['收盘价'][i + N0] / data['成交量'][i + N0]
                        new_gpdata['20maxd'][i] = (data['收盘价'][i + N0] * data['成交量'][i + N0] - data['ma20d'][i + N0]) / \
                                                  data['收盘价'][i + N0] / data['成交量'][i + N0]
                        new_gpdata['30maxd'][i] = (data['收盘价'][i + N0] * data['成交量'][i + N0] - data['ma30d'][i + N0]) / \
                                                  data['收盘价'][i + N0] / data['成交量'][i + N0]
                        new_gpdata['60maxd'][i] = (data['收盘价'][i + N0] * data['成交量'][i + N0] - data['ma60d'][i + N0]) / \
                                                  data['收盘价'][i + N0] / data['成交量'][i + N0]
                        new_gpdata['120maxd'][i] = (data['收盘价'][i + N0] * data['成交量'][i + N0] - data['ma120d'][
                            i + N0]) / data['收盘价'][i + N0] / data['成交量'][i + N0]
                        new_gpdata['250maxd'][i] = (data['收盘价'][i + N0] * data['成交量'][i + N0] - data['ma250d'][
                            i + N0]) / data['收盘价'][i + N0] / data['成交量'][i + N0]
                        new_gpdata['MACD'][i] = data['MACD'][i + N0]
                        new_gpdata['DEA'][i] = data['DEA'][i + N0]
                        new_gpdata['SKDJ'][i] = data['SKDJ'][i + N0]
                        new_gpdata['LSKDJ'][i] = data['LSKDJ'][i + N0]
                        new_gpdata['RSI'][i] = data['RSI'][i + N0]

                    for i in range(0, len(gpdata) - N - PRE):
                        new_gpdata['gp price'][i] = (gpdata['收盘价'][i + N] - gpdata['收盘价'][i + N - 1]) / gpdata['收盘价'][
                            i + N - 1]
                        new_gpdata['gp 5ma'][i] = (gpdata['收盘价'][i + N] - gpdata['ma5'][i + N]) / gpdata['收盘价'][i + N]
                        new_gpdata['gp 10ma'][i] = (gpdata['收盘价'][i + N] - gpdata['ma10'][i + N]) / gpdata['收盘价'][i + N]
                        new_gpdata['gp 20ma'][i] = (gpdata['收盘价'][i + N] - gpdata['ma20'][i + N]) / gpdata['收盘价'][i + N]
                        new_gpdata['gp 30ma'][i] = (gpdata['收盘价'][i + N] - gpdata['ma30'][i + N]) / gpdata['收盘价'][i + N]
                        new_gpdata['gp 60ma'][i] = (gpdata['收盘价'][i + N] - gpdata['ma60'][i + N]) / gpdata['收盘价'][i + N]
                        new_gpdata['gp 120ma'][i] = (gpdata['收盘价'][i + N] - gpdata['ma120'][i + N]) / gpdata['收盘价'][
                            i + N]
                        new_gpdata['gp 250ma'][i] = (gpdata['收盘价'][i + N] - gpdata['ma250'][i + N]) / gpdata['收盘价'][
                            i + N]
                        # print('end gp price',new_gpdata['end gp price'][i])
                        new_gpdata['gp upper_band'][i] = (gpdata['upper_band'][i + N] - gpdata['收盘价'][i + N]) / \
                                                         gpdata['收盘价'][i + N]
                        # print('high gp price',new_gpdata['high gp price'][i])
                        new_gpdata['gp lower_band'][i] = (gpdata['收盘价'][i + N] - gpdata['lower_band'][i + N]) / \
                                                         gpdata['收盘价'][i + N]
                        # print(i + N - 1, gpdata['成交量'][i + N - 1])
                        new_gpdata['gp amount'][i] = (float(gpdata['成交量'][i + N]) - float(
                            gpdata['成交量'][i + N - 1])) / float(
                            gpdata['成交量'][i + N - 1])
                        new_gpdata['gp 5mad'][i] = (gpdata['成交量'][i + N] - gpdata['mad5'][i + N]) / gpdata['成交量'][i + N]
                        new_gpdata['gp 10mad'][i] = (gpdata['成交量'][i + N] - gpdata['mad10'][i + N]) / gpdata['成交量'][
                            i + N]
                        new_gpdata['gp 20mad'][i] = (gpdata['成交量'][i + N] - gpdata['mad20'][i + N]) / gpdata['成交量'][
                            i + N]
                        new_gpdata['gp 30mad'][i] = (gpdata['成交量'][i + N] - gpdata['mad30'][i + N]) / gpdata['成交量'][
                            i + N]
                        new_gpdata['gp 60mad'][i] = (gpdata['成交量'][i + N] - gpdata['mad60'][i + N]) / gpdata['成交量'][
                            i + N]

                        new_gpdata['gp pricemad'][i] = (gpdata['收盘价'][i + N] * gpdata['成交量'][i + N] - gpdata['收盘价'][
                            i + N - 1] *
                                                        gpdata['成交量'][i + N - 1]) / gpdata['收盘价'][i + N - 1] / \
                                                       gpdata['成交量'][i + N - 1]
                        new_gpdata['gp 5maxd'][i] = (gpdata['收盘价'][i + N] * gpdata['成交量'][i + N] - gpdata['ma5d'][
                            i + N]) / \
                                                    gpdata['收盘价'][i + N] / gpdata['成交量'][i + N]
                        new_gpdata['gp 10maxd'][i] = (gpdata['收盘价'][i + N] * gpdata['成交量'][i + N] - gpdata['ma10d'][
                            i + N]) / \
                                                     gpdata['收盘价'][i + N] / gpdata['成交量'][i + N]
                        new_gpdata['gp 20maxd'][i] = (gpdata['收盘价'][i + N] * gpdata['成交量'][i + N] - gpdata['ma20d'][
                            i + N]) / \
                                                     gpdata['收盘价'][i + N] / gpdata['成交量'][i + N]
                        new_gpdata['gp 30maxd'][i] = (gpdata['收盘价'][i + N] * gpdata['成交量'][i + N] - gpdata['ma30d'][
                            i + N]) / \
                                                     gpdata['收盘价'][i + N] / gpdata['成交量'][i + N]
                        new_gpdata['gp 60maxd'][i] = (gpdata['收盘价'][i + N] * gpdata['成交量'][i + N] - gpdata['ma60d'][
                            i + N]) / \
                                                     gpdata['收盘价'][i + N] / gpdata['成交量'][i + N]
                        new_gpdata['gp 120maxd'][i] = (gpdata['收盘价'][i + N] * gpdata['成交量'][i + N] - gpdata['ma120d'][
                            i + N]) / \
                                                      gpdata['收盘价'][i + N] / gpdata['成交量'][i + N]
                        new_gpdata['gp 250maxd'][i] = (gpdata['收盘价'][i + N] * gpdata['成交量'][i + N] - gpdata['ma250d'][
                            i + N]) / \
                                                      gpdata['收盘价'][i + N] / gpdata['成交量'][i + N]

                        new_gpdata['gp MACD'][i] = gpdata['MACD'][i + N]
                        new_gpdata['gp DEA'][i] = gpdata['DEA'][i + N]
                        new_gpdata['gp SKDJ'][i] = gpdata['SKDJ'][i + N]
                        new_gpdata['gp LSKDJ'][i] = gpdata['LSKDJ'][i + N]
                        new_gpdata['gp RSI'][i] = gpdata['RSI'][i + N]

                    # label_data = pd.DataFrame(index=range(0, len(gpdata) - N - PRE - 1), columns=['growth rate'])
                    tmp_y = []

                    for i in range(0, len(gpdata) - N):
                        # print(gpdata['日期'][i + N],(max(gpdata['最高价'][i + N + 1 - PRE:i + N + 1]) - gpdata['收盘价'][i + N - PRE])/ gpdata['收盘价'][i + N - PRE]*100,min(gpdata['最低价'][i + N + 1 - PRE:i + N + 1]) ,gpdata['收盘价'][i + N - PRE])
                        tmp_y.append(
                            [(max(gpdata['最高价'][i + N + 1 - PRE:i + N + 1]) - gpdata['收盘价'][i + N - PRE]) /
                             gpdata['收盘价'][i + N - PRE] * 100
                                , (min(gpdata['最低价'][i + N + 1 - PRE:i + N + 1]) - gpdata['收盘价'][i + N - PRE]) /
                             gpdata['收盘价'][i + N - PRE] * 100])

                    new_gpdata.index = new_gpdata['gp date']

                    new_gpdata.drop('gp date', axis=1, inplace=True)

                    # print(new_gpdata.isna().any())

                    dataset = new_gpdata.values

                    train = dataset[:, :]
                    # valid = dataset[1900:, :]
                    scalerx = MinMaxScaler(feature_range=(0, 1))

                    scaled_data = scalerx.fit_transform(dataset)

                    # x个参数 INPUT_N个数据组作为预估
                    for i in range(INPUT_N, len(train) - PRE):
                        x_data.append(scaled_data[i - INPUT_N: i])
                        # print('x_train',x_train[i - 60])
                        y_data.append(tmp_y[i + PRE])


                else:
                    print("bad pre data")
                    exit(0)

                x_data, y_data = np.array(x_data), np.array(y_data)

                x_data = np.reshape(x_data, (x_data.shape[0], x_data.shape[1], x_data.shape[2]))
                # print(x_data, x_data.shape[0], x_data.shape[1], x_data.shape[2])

                y_data = np.reshape(y_data, (y_data.shape[0], y_data.shape[1]))
                print(y_data)

                x_pre = []
                x_pre.append(scaled_data[len(train) - INPUT_N - TESTREV: len(train) - TESTREV])
                #print(scaled_data)

                build_LSTM_model(x_data, y_data, x_pre, B_TRAIN, code)

def main():
    #download_gjcsv('./auto_download_stockdata/')
    stock_pre('./auto_download_stockdata/')


if __name__ == '__main__':
    main()
    time.sleep(1)
