# -*- coding: utf-8 -*-
'''
* 本スクリプトの処理の内容
本スクリプトは、ニューロン単体が四則演算だけで正解率100%にたどり着く過程が記述されているスクリプト。


* コードの見方
本スクリプトを実行した時に呼び出されるメソッドはmainメソッドなので、そこから処理を追うのがよい。
'''

# -*- coding: utf-8 -*-

INPUT_VALUES = [
  [0, 0], # ジャンプではない & 海賊漫画でもない
  [1, 0], # ジャンプである　 & 海賊漫画でもない
  [0, 1], # ジャンプではない & 海賊漫画である
  [1, 1], # ジャンプである　 & 海賊漫画である
]

REAL_ANSWERS = [
  0, # ワンピースではない
  0, # ワンピースではない
  0, # ワンピースではない
  1  # ワンピースである
]

'''ステップ関数'''
def step_function(v):
    if v >= 0:
        return 1
    else:
        return 0


'''入力値を与えて結果を予想'''
def predict(x1, x2, w1, w2, b):
    return step_function(x1 * w1 + x2 * w2 - b)


'''学習'''
def learn(x1, x2, w1, w2, b, r, predict_answer, real_answer):
    diff = real_answer - predict_answer # 出力値と実際の値の差
    wr1 = x1 * w1 * diff * r
    wr2 = x2 * w2 * diff * r
    w1 = w1 + wr1 # 重み1を更新
    w2 = w2 + wr2 # 重み2を更新

    br = diff * r
    b = b - br # バイアス（閾値）を更新

    return w1, w2, b # 更新した値を返却


'''main処理'''
def main():
    w1 = 2  # 重み1
    w2 = 1 # 重み2
    b = -2   # バイアス（閾値）
    r = 0.5 # 更新率

    correct_counts = 0 # 正解回数
    learning_count = 0 # 学習回数
    loop = 0

    print("----- 学習開始 -----")
    while correct_counts < 4: # 4つ全てが正解になるまで学習を続ける。
        index = loop % 4
        x1 = INPUT_VALUES[index][0] # 入力値 X1
        x2 = INPUT_VALUES[index][1] # 入力値 X2

        # 予想結果
        predict_answer = predict(x1, x2, w1, w2, b)
        # 実際の結果
        real_answer = REAL_ANSWERS[index]

        if predict_answer == real_answer:
            correct_counts += 1 # 正解！！
        else:
            correct_counts = 0 # 残念...
            learning_count += 1
            w1, w2, b = learn(x1, x2, w1, w2, b, r, predict_answer, real_answer) # 学習
            print("  学習して重みとバイアスが更新された！！(" + str(learning_count) + "度目の学習)")
            print("  w1 : " + str(w1) + " , w2 : " + str(w2) + " , b : " + str(b))
        loop += 1

    print("学習完了！！")
    print("学習回数 : " + str(learning_count))
    print("最終的な重みとバイアスの値は以下の通り。")
    print("w1 : " + str(w1) + " , w2 : " + str(w2) + " , b : " + str(b))

if __name__ == '__main__':
    main()
