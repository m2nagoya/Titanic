# -*- coding: utf-8 -*-

import pandas as pd
import numpy  as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

## CSV読み込み
train = pd.read_csv("./titanic/train.csv")
test  = pd.read_csv("./titanic/test.csv")

### セル表示
# train ## 全体
# train.head(10) ## 先頭から10行目
# train.describe() ## 統計量
# train.info() ## 要約情報
# train['Age'].isnull().values.sum() ## nullの件数

### ヒストグラム
# plt.hist(train['Age'].dropna(), bins=20)

### 欠損値補完 (NULL -> AVG)
mean = np.mean(train['Age'])
train['Age'] = train['Age'].fillna(mean)

### 定量化 (男 -> 1, 女 -> 2)
train['Sex'] = train['Sex'].str.replace('female', '2')
train['Sex'] = train['Sex'].str.replace('male', '1')

### データセットの作成 (説明変数 -> X, 目的変数 -> Y)
X = pd.DataFrame({'Pclass':train['Pclass'], 'Sex':train['Sex'], 'Age':train['Age']})
y = pd.DataFrame({'Survived':train['Survived']})

### データセットの分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=None)

### 学習
model = SVC(kernel='linear', random_state=None,C=0.1)
model.fit(X_train, y_train)

### 性能評価
model.score(X_test, y_test)
