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


import numpy as np

HAND_G = 0
HAND_C = 1
HAND_P = 2
HAND_NAMES = ["グー", "チョキ", "パー"]

HAND_G_ONE_HOT = [1, 0, 0]
HAND_C_ONE_HOT = [0, 1, 0]
HAND_P_ONE_HOT = [0, 0, 1]

WIN_HAND = [HAND_P_ONE_HOT, HAND_G_ONE_HOT, HAND_C_ONE_HOT]

# 学習率
LEARNING_RATE = 0.0005

'''シグモイド関数（活性化関数の一種）'''
# 活性化関数の一種。
# 1 / 1 + exp(-a)
# 参考資料 : https://mathtrain.jp/sigmoid
#
# @param  : numpy.ndarray
# @return : numpy.ndarray
def sigmoid(layer_values):
    return 1 / (1 + np.exp(-layer_values))


'''ソフトマックス関数'''
# 配列全てのあたいの合計が１になるようにする関数。これにより出力結果を確率で示せるようになる。
# exp(a) / Σn exp(a)
#
# @param  : numpy.ndarray
# @return : numpy.ndarray
def softmax(layer_values):
    max_value = np.max(layer_values)

    # ソフトマックス関数をプログラミングでそのまま表現すると一定以上でオーバーフローするので、値を抑えるように式を修正している。(一般的なやり方)
    exp_a = np.exp(layer_values - max_value);
    sum_exp_a = np.sum(exp_a)

    return exp_a / sum_exp_a


'''順伝播(入力値から出力値を予想)'''
# @return y  : numpy.ndarray ３層目(出力)の値
# @return z1 : numpy.ndarray 2層目の値
def forward_propagation(x, W1, W2):
    u1 = x.dot(W1)   # 1層→2層目の重みをかけている。
    z1 = sigmoid(u1) # 活性化関数(シグモイド関数)を適用
    u2 = z1.dot(W2)  # 2層→3層目の重みをかけている。
    y = softmax(u2)  # u2の合計値が1になるようにしている。
    return y, z1


'''誤差逆伝播(学習)'''
'''
x  : 学習データ(入力値)
z1 : ２層目の各値
y  : 学習の予想結果
d  : 教師データの結果
W1 : 1→2層目の重み
W2 : 2→3層目の重み
'''
def back_propagation(x, z1, y, d, W1, W2):
    delta2 = y - d
    grad_W2 = z1.T.dot(delta2)

    sigmoid_dash = z1 * (1 - z1)
    delta1 = delta2.dot(W2.T) * sigmoid_dash
    grad_W1 = x.T.dot(delta1)

    W2 -= LEARNING_RATE * grad_W2
    W1 -= LEARNING_RATE * grad_W1

    print(LEARNING_RATE * grad_W1)
    print(W2)
    print(W2.T)

    return W1, W2


'''学習データを与えて、モデルを学習'''
def learn(train_X, train_Y, W1, W2):
    for train_x, train_y in zip(train_X, train_Y):
        # 順伝播
        x = np.array(train_x)
        y, z1 = forward_propagation(x, W1, W2)

        # 誤差逆伝播
        d = np.array(train_y) # 教師データ
        W1, W2 = back_propagation(x, z1, y, d, W1, W2)


'''モデルにじゃんけんをさせる。手を予想'''
def predict(hand, W1, W2):
    y, _ = forward_propagation(np.array(hand), W1, W2)
    print(y)
    print("君「" + HAND_NAMES[hand] + "には何を出せば勝てる？」")
    print("AI「" + HAND_NAMES[y.argmax()] + "を出せば勝てるよ！」")


'''重みの初期値を取得'''
def get_initial_weights():
    W1 = np.array([0.1, 0.2, 0.1], dtype = 'float64') # 1層目→2層目の重み
    W2 = np.array([[-0.1, 0.3, 0.1], [0.2, 0.1, 0.3], [0.1, 0.5, 0.1]], dtype = 'float64') # 2層目→3層目の重み
    return W1, W2

'''
各層のニューロンの数
１層目(入力) : 1（0~2の値が入る。それぞれグー・チョキ・パーを示す。）
２層目　　   : 3
３層目(出力) : 3（各ニューロンに0~1の値が出力され、合計値が1になる。値は確率を示す。）
'''

'''main処理'''
def main():
    # 今回使う重み
    # W1 : 1層目→2層目の重み
    # W2 : 2層目→3層目の重み
    W1, W2 = get_initial_weights()

    print("学習前のAI君とのやりとり")
    predict(HAND_G, W1, W2)
    predict(HAND_C, W1, W2)
    predict(HAND_P, W1, W2)
    print(W1)
    print(W2)

    train_x = np.array([HAND_G, HAND_C, HAND_P])
    train_y = np.array([WIN_HAND[HAND_G], WIN_HAND[HAND_C], WIN_HAND[HAND_P]])

    for _ in range(15):
        learn(train_x, train_y, W1, W2)

    print("\n学習後のAI君とのやりとり")
    predict(HAND_G, W1, W2)
    predict(HAND_C, W1, W2)
    predict(HAND_P, W1, W2)
    print(W1)
    print(W2)

if __name__ == '__main__':
      main()
