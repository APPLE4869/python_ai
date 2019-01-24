# -*- coding: utf-8 -*-
'''
* 本スクリプトの処理の内容
本スクリプトは、Deep Learning界のHello Worldと言われるMNISTの処理を記述したスクリプト。
たったこれだけのコードで正答率98%を越える10年前では考えられない数字を叩き出せる。


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

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout
from keras.optimizers import RMSprop
from keras.utils.np_utils import to_categorical # データをone hot表現に整形するのに利用。

# MNISTで利用する数字データが実際にどんなものなのか見たい際はTrueにすると、実行後最初に出力される。
PRINT_NUM_DATA_EXAMPLE = False

BATCH_SIZE = 128
EPOCHS = 20

'''Kerasが用意してくれているmnist用のデータを整形して取得'''
def get_mnist_data():
    # Kerasがデータを用意してくれているので、mnist.load_data()だけで、対象のデータを用意できる。
    # 参考資料 : https://keras.io/ja/datasets/#mnist
    #
    # tran_x : 60,000枚の手書き数字画像のデータ（画像は28 x 28の配列で各マスの濃さを0~255で表している。）
    # tran_y : tran_xの各手書き画像の数字を数値として持っている。
    # test_x : 10,000枚の手書き数字画像のデータ（画像は28 x 28の配列で各マスの濃さを0~255で表している。）
    # test_y : text_xの各手書き画像の数字を数値として持っている。
    (train_x, train_y), (test_x, test_y) = mnist.load_data()

    if PRINT_NUM_DATA_EXAMPLE: # 数字データがどんなものかイメージがつかない場合は、定数をTrueにすると出力される。
        print("# 数字データ例")
        print("## 数字データ")
        print(train_x[0])
        print("## 数字データが示す数字")
        print(train_y[0])

    train_x = train_x.reshape(60000, 784) # 28 x 28で表現されている数字画像を784 x 1に変換
    test_x = test_x.reshape(10000, 784)   # 28 x 28で表現されている数字画像を784 x 1に変換
    train_x = train_x.astype('float32')   # int型をfloat32型に変換
    test_x = test_x.astype('float32')     # int型をfloat32型に変換 
    train_x /= 255 # 数字画像のマスの濃さを[0-255]で表現していたものを[0.0-1.0]に変換
    test_x /= 255  # 数字画像のマスの濃さを[0-255]で表現していたものを[0.0-1.0]に変換

    # * to_categoricalメソッド
    # 説明    : 数字の結果をone hot表現に整形
    # 参考資料 : 参考資料 : https://keras.io/ja/utils/#to_categorical
    #
    # 例
    # 変換前 → 変換後
    # 0 → [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    # 3 → [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    # 9 → [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    train_y = to_categorical(train_y, 10) # 今回はtran_yの値の種類が必ず0~9の10種類になるので、第二引数に10と入れている。
    test_y = to_categorical(test_y, 10)   # 今回はtran_yの値の種類が必ず0~9の10種類になるので、第二引数に10と入れている。

    return train_x, train_y, test_x, test_y


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
    #
    # * Dropoutクラス
    # 説明    : 訓練時の更新においてランダムに入力ユニットを0とする割合であり，過学習の防止に役だつ。
    # 参考資料 : https://keras.io/ja/layers/core/#dropout
    model = Sequential()
    model.add(Dense(512, input_shape=(784,))) # 今回の入力値は784 x nであるため、input_shapeはこのように指定している。
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(10)) # 最終的に画像データが0~9の10種類のどれを示すものなのかを確率で示すため、出力する数を10にしている。
    model.add(Activation('softmax'))

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
    model.compile(optimizer=RMSprop(), loss='categorical_crossentropy', metrics=['accuracy'])

    return model


'''モデルを学習させる'''
def learn_by_using_data(model, train_x, train_y, test_x, test_y):
    model.fit(train_x, train_y,  # 画像とラベルデータ
              batch_size=BATCH_SIZE,
              epochs=EPOCHS,     # エポック数の指定
              validation_data=(test_x, test_y))


'''正解率を出力する'''
def print_correct_answer_rate(model, test_x, test_y):
    score = model.evaluate(test_x, test_y, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


'''main処理'''
def main():
    # 過去のデータ一番最後のデータを取得
    # train_x    : 学習用の複数の数字画像データ(多次元配列)
    # train_y    : 学習用の数字画像データそれぞれが示す数字（答え）
    # test_x     : テスト用の複数の数字画像データ(多次元配列)
    # test_y     : テスト用の数字画像データそれぞれが示す数字（答え）
    train_x, train_y, test_x, test_y = get_mnist_data()

    # 上で取得したデータを利用して学習
    model = get_neural_network_model()
    learn_by_using_data(model, train_x, train_y, test_x, test_y)

    # 学習したmodelを利用して正解率を出力。
    print_correct_answer_rate(model, test_x, test_y)

if __name__ == '__main__':
    main()
