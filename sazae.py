# -*- coding: utf-8 -*-
'''
* 本スクリプトの処理の内容
本スクリプトは、サザエさんが来週じゃんけんで出す手を予想する処理が記述されているスクリプト。
Deep LearningフレームワークであるKerasを利用して sazae_data.tsv にある過去のサザエさんが出した手を元に、来週サザエさんが出す手を予想する。


* Kerasとは？
Googleが提供するDeep Learningフレームワーク Tensorflow上で動くDeep Learningフレームワーク。
Deep Learningフレームワークの中では２番目(Tensorflowの次)に多いスター数を誇る。(2018/07/17時点)
Deep Learningを行う際に必要になる行列計算を始めとした様々な計算処理をクラス化・メソッド化して、
開発者が本当にやるべきロジックのみに専念して機械学習の開発をすることができるようになる。


* コードの見方
前提として、本スクリプトを実行した時に呼び出されるメソッドはmainメソッドのみ。
なので、mainメソッドの処理でどのような流れで機械学習が行われているのかの概要を把握し、
その後、切り出されたメソッドで細かい処理の流れを1つずつ理解していくのが良さそう。
一応、各処理の内容や目的等はコメントで書いておいたので、何も調べないでもおおよそは理解できるはず。


* 注意点！
pythonでは、関数の引数に変数を渡した場合、全て(値渡しではなく)参照渡しになる仕様みたいなので、以下のコードはそのことに注意。
'''

import numpy as np # 行列計算や多次元配列を容易に扱うためのライブラリ。今回は主にKerasのメソッドに渡すためのデータを整形するのに利用。
import pandas as pd # CSVを始めとした様々なデータを扱うためのライブラリ。今回はTSVのデータを読み込むのに利用。
from keras.layers import Activation, Dense # Kerasのクラス。ニューラルネットワークのレイヤー構築に利用。
from keras.models import Sequential # Kerasのクラス。ニューラルネットワークのモデルを構築するために利用。

data_file = 'sazae_data.tsv'

'''リスト(配列)をまとめてシャッフルする'''
def shuffle_lists(list1, list2):
    seed = np.random.randint(0, 1000)
    np.random.seed(seed)
    np.random.shuffle(list1)
    np.random.seed(seed)
    np.random.shuffle(list2)


'''TSVファイルからサザエさんが出した手のデータを整形して取得する'''
'''
* データの整形方式
1週ごとに出された手以外に1を足して、出された手は0で初期化していく方式でデータを整形。
このデータをどう扱うかがDeep Learningでより良い結果を立つためには最重要になってくる。

例(左からグー・チョキ・パー)
1週目 : [0, 0, 0]
2週目 : [1, 0, 1] ← 1週目でチョキが出された
3週目 : [0, 1, 2] ← 2週目でグーが出された
4週目 : [0, 2, 3] ← 3週目でグーが出された
'''
def get_data_from_sazae_tsv():
    zyanken_data = pd.read_csv(data_file, sep='\t', usecols=['rock', 'scissors', 'paper'])
    x_data = [[0, 0, 0]] # 1週目のデータ(ここに整形した配列データがどんどん追加されていく。)
    for row in zyanken_data.values:
        data = x_data[-1] # 最週のデータ配列を取得
        data = list(map(lambda x: x + 1, data)) # 配列の値全てに1を足す
        data[row.argmax()] = 0 # 前回出た手を0で初期化
        x_data.append(data)

    # numpy.array型に変換
    # x_dataとy_dataで取得する位置が1つずれているのは、x(出ている手の状態)の影響が作用するのは次週のy(手の結果)であるため、1つずれている。
    X_data = np.array(x_data[-501:-1]) # 最後から501番目 ~ 最後から２番目
    Y_data = np.array(zyanken_data.values[-500:]) # 最後から500番目以降。じゃんけんの結果。

    last_data = np.array(X_data[-1:]) # 一番最後の手の状況（次週の手を予想する際に利用する。）

    # 正規化(配列内の数値がint64なので、計算用にfloat32にキャスト)
    X_data = X_data.astype(np.float32)
    last_data = last_data.astype(np.float32)

    # シャッフル（データの偏りで局所解にハマるリスクを軽減するために、配列をシャッフルしている。）
    # 参考資料 : https://qiita.com/koshian2/items/028c457880c0ec576e27
    shuffle_lists(X_data, Y_data)

    return X_data, Y_data, last_data


'''モデル(ニューラルネットワーク)を構築する'''
def get_neural_network_model():
    # * Keras Document（Kerasの基本的な使い方はは以下のドキュメントを読めばおおよそわかる。）
    # 参考資料 : https://keras.io/ja/

    # * Sequentialモデル
    # 説明 : よくみるニューラルネットワークを構築するために利用されるクラス。
    #       このインスタンスのメソッドでニューロンの数などを簡単に設定することができる。
    # 参考資料 : https://keras.io/ja/getting-started/sequential-model-guide/
    #
    #
    # * addメソッド
    # 説明 : モデルに対して、レイヤーや活性化関数の指定を追加するメソッド。
    #       引数にはDenseやActivationのインスタンスが入る。
    #
    #
    # * Denseクラス
    # 説明    : レイヤーを追加するためのクラス。第一引数でニューロンの数を指定できる。
    # 参考資料 : https://keras.io/ja/layers/core/#activation
    #
    # * Activationクラス
    # 説明    : 活性化関数の指定をするためのクラス。メジャーどころだと、 relu, sigmoid, softmaxなどが引数に入る。
    # 参考資料 : https://keras.io/ja/activations/
    model = Sequential()
    model.add(Dense(32, input_shape=(3,))) # 今回の入力値はグー・チョキ・パーの3種類なので入力値を3つに指定。
    model.add(Activation('sigmoid'))
    model.add(Dense(32))
    model.add(Activation('sigmoid'))
    model.add(Dense(32))
    model.add(Activation('relu'))
    model.add(Dense(3))
    model.add(Activation('softmax')) # グー・チョキ・パーの出力値の合計が1になるようになっている。1なのは0.23 = 23%などを表し、出る手の確率を表現するため。

    # * compileメソッド
    # 説明 : 入力値を元にどのように学習するかを諸々設定するメソッド。専門知識も必要なので、入門では特に意識せずdefaultの値を適当に入れるのでも良い。
    #       より精度の高い学習を行いたくなった時に、以下の引数をいじるようにすればいい。
    # 参考資料 : https://keras.io/ja/models/model/#compile
    #
    # * 今回利用した引数
    # optimizer : 勾配法を指定。
    # 参考資料   : https://qiita.com/tokkuman/items/1944c00415d129ca0ee9
    #             https://keras.io/ja/optimizers/
    #
    # loss    : 損失関数を指定。
    # 参考資料 : https://keras.io/ja/losses/
    #
    # metrics : 評価関数を指定
    # 参考資料 : https://keras.io/ja/metrics/
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model


'''モデルを学習させる'''
def learn_by_using_data(model, train_x, train_y, test_x, test_y):
      # * fitメソッド
      # 説明    : モデルを学習させるメソッド。試行回数など諸々の値を設定できる。
      # 参考資料 : https://keras.io/ja/models/model/#fit
      model.fit(train_x, train_y,
                epochs=1000,
                batch_size=16,
                validation_data=(test_x, test_y),
                shuffle=True)


'''TSVから取得したデータを学習用とテスト用に分割する'''
def split_into_train_and_test_data(x_data, y_data):
    # 前半8割のデータを学習用、後半2割のデータをテスト用に利用する。
    mid = int(len(y_data) * 0.8)
    train_x, train_y = x_data[:mid], y_data[:mid]
    test_x, test_y = x_data[mid:], y_data[mid:]
    return train_x, train_y, test_x, test_y

'''正解率を出力する'''
def print_correct_answer_rate(model, X, Y, label):
    predictX = model.predict(X)
    correct = 0
    # zip関数
    # 説明    : pythonの標準関数。複数のリスト(配列)を同時にイテレートできる。
    # 参考資料 : https://note.nkmk.me/python-zip-usage-for/
    for real, predict in zip(Y, predictX):
        if real.argmax() == predict.argmax(): # 予想(最も確率の高い手)と結果が一致した場合は、+1。
            correct += 1
    correct = correct / len(Y)
    print(label + '正解率 : %02.2f ' % correct)


'''来週の手を予想する'''
def print_next_hand(model, last_data):
    # * predictメソッド
    # 説明    : 引数に渡した入力値に対しての予測結果が返される。
    # 参考資料 : https://keras.io/ja/models/model/#predict
    next_hand = model.predict(last_data) # 
    hands = ['グー', 'チョキ', 'パー']
    print('来週の手 : ' + hands[next_hand[0].argmax()]) # 最も出る確率の高かった手を出力
    print('(グー: ' + str(next_hand[0][0]) + ', チョキ:' + str(next_hand[0][1]) + ', パー:' + str(next_hand[0][2]) + ')')


'''main処理'''
def main():
    # 過去のデータ一番最後のデータを取得
    # x_data    : 前の週の出た手の状況(配列で複数週分のデータが格納されている。)
    # y_data    : 次週に出た手の結果(配列で複数週分のデータが格納されている。)
    # last_data : 一番最後の手の状況(来週の手を予想するための)
    x_data, y_data, last_data = get_data_from_sazae_tsv()

    # 取得したデータを学習用とテスト用に分割
    # 学習用のデータでモデルを学習させ、未学習のテスト用のデータで学習がうまくいっているかを確認する。
    train_x, train_y, test_x, test_y = split_into_train_and_test_data(x_data, y_data)

    # 上で取得したデータを利用して学習
    model = get_neural_network_model()
    learn_by_using_data(model, train_x, train_y, test_x, test_y)

    # 学習したmodelを利用して正解率を出力。
    # 学習データとテストデータでの正答率が大きく乖離している状態が、いわゆる過学習の状態になる。
    print_correct_answer_rate(model, train_x, train_y, 'train') # 学習データでの正答率を出力
    print_correct_answer_rate(model, test_x, test_y, 'test')    # テストデータでの正答率を出力

    # 来週の手を予想し出力
    print_next_hand(model, last_data)

if __name__ == '__main__':
    main()
