# -*- coding: utf-8 -*-
'''
* 本スクリプトの処理の内容
本スクリプトは、ディープラーニングでじゃんけんに勝てるAIを作成するコードが記述されたスクリプト。


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
from keras.models import Sequential # Kerasのクラス。ニューラルネットワークのモデルを構築するために利用。
from keras.layers import Dense, Activation # Kerasのクラス。ニューラルネットワークのレイヤー構築に利用。
from keras.utils.np_utils import to_categorical # データをone hot表現に整形するのに利用。


'''変更すると結果が変わる遊べる定数'''
# 一つの訓練データを何回繰り返して学習させるか
EPOCHS = 200
# 教師データの増やす倍率（じゃんけんの手の出方は9パターンしかなく、それだけだと学習に偏りが出るので、９パターンのデータを一定率増やして学習させている。）
REPEAT_CNT = 100
# モデルの結果予想の詳細(確率の内容)を表示するかどうか
PRINT_DETAIL_RATE = False
# 学習したモデルの重みとバイアスを表示するかどうか
PRINT_WEIGHT_AND_BIAS = False


'''基本的には変更してはいけない定数'''
# じゃんけんの手
HAND_G = 0
HAND_C = 1
HAND_P = 2
HAND_SET = [HAND_G, HAND_C, HAND_P]
JP_HAND_NAMES = ["グー　", "チョキ", "パー　"] # 最後の出力結果で見やすいように全角スペースを入れて同じ文字数にしている。

# じゃんけんの手とその結果。勝敗は左の手を基準にしている。（合計９パターン）
WIN_HAND = [HAND_P, HAND_G, HAND_C]


'''学習用データを取得する'''
def get_training_data():
    # 教師データ じゃんけんの手の組み合わせ
    #
    # * repeatメソッド
    # 説明    : 配列を一定回数分複製する。データ量が少ない時などは学習時に偏りが発生しやすくなるので、データ量を増やして対策したりする。
    # 参考資料 : https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.repeat.html
    training_data = np.array(HAND_SET).repeat(REPEAT_CNT, axis=0)

    # 教師データ じゃんけんの手に合わせた勝敗
    label = np.array(WIN_HAND).repeat(REPEAT_CNT)

    # * to_categoricalメソッド
    # 説明    : もともと 0 or 1 or 2 であるじゃんけんの手のデータを以下の形式に変換する。（いわゆるone hot表現）
    #          この形式はどの結果である確率が何%かなどを示すのにとても有効なので、機械学習ではかなり重宝されている。
    #          グー　 [1,0,0]
    #          チョキ [0,1,0]
    #          パー   [0,0,1]
    # 参考資料 : https://keras.io/ja/utils/#to_categorical
    label_one_hot = to_categorical(label)

    return training_data, label_one_hot


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
    model.add(Dense(10, activation='relu', input_dim=1)) # 今回の入力値は自分の手と相手の手の2種類なので入力値を2つに指定。
    model.add(Dense(3, activation='softmax')) # 引分・勝利・敗北の出力値の合計が1になるようになっている。1なのは0.23 = 23%などを表し、出る手の確率を表現するため。

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
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    return model


'''モデルを学習させる'''
def learn_by_using_data(model, training_data, label):
    # * fitメソッド
    # 説明    : モデルを学習させるメソッド。試行回数など諸々の値を設定できる。
    # 参考資料 : https://keras.io/ja/models/model/#fit
    model.fit(training_data, label, epochs=EPOCHS, batch_size=30)


'''じゃんけんの結果を予想し、その正解率を出力する'''
def print_correct_answer_rate(model) :
    correct = 0

    print("*----- 結果 -----*")
    print("出された手 => AIが出した手 : 勝てた？")

    # zip関数
    # 説明    : pythonの標準関数。複数のリスト(配列)を同時にイテレートできる。
    # 参考資料 : https://note.nkmk.me/python-zip-usage-for/
    for hand, real in zip(HAND_SET, WIN_HAND):
      predict = model.predict(np.array([hand])) # 自分と相手の手を入力してモデルに結果を予想してもらう。

      predict_result = np.argmax(predict) # 予想結果の数値を取得。(引分 : 0, 勝利 : 1, 敗北 : 2)

      if predict_result == real:
        correct += 1

      print(JP_HAND_NAMES[hand] + ' => ' + JP_HAND_NAMES[predict_result] + ' : ' + str(predict_result == real))

      # モデルが勝率をどのように予測したのかの内訳を出力
      if PRINT_DETAIL_RATE:
        print(' ↑予想確率詳細 : グー->%02.2f' % predict[0][0] + ' チョキ->%02.2f' % predict[0][1] + ' パー->%02.2f' % predict[0][2])

    print("*----- 結果 -----*")

    correct = correct / len(HAND_SET) * 100
    print('正解率 : %02.2f ' % correct + '%')


'''学習したモデルがもつ重みやバイアスを出力'''
def print_weights(model):
    weights = model.get_weights()
    print("-----------------------------------------")
    print("# １層→２層の重み　　 10(出力値の数) x 1(入力値の数)")
    print(weights[0])
    print("# １層→２層のバイアス 10(出力値の数) x 1(入力値の数)")
    print(weights[1])
    print("# ２層→３層の重み　　 3(出力値の数) x 10(入力値の数)")
    print(weights[2])
    print("# ２層→３層のバイアス 3(出力値の数) x 10(入力値の数)")
    print(weights[3])


'''main処理'''
def main():
    # 過去のデータ一番最後のデータを取得
    # training_data : 学習用のデータ
    # label         : 結果
    training_data, label = get_training_data()

    # 上で取得したデータを利用して学習
    model = get_neural_network_model()
    learn_by_using_data(model, training_data, label)

    # 学習したmodelを利用して正解率を出力
    print_correct_answer_rate(model)

    # 値がTrueの時は学習したモデルの重みとバイアスを出力
    if PRINT_WEIGHT_AND_BIAS:
        print_weights(model)

if __name__ == '__main__':
    main()
