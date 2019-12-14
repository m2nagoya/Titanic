# -*- coding: utf-8 -*-

import pandas as pd
import numpy  as np
import matplotlib.pyplot as plt

## CSV読み込み
train = pd.read_csv("./titanic/train.csv")
test  = pd.read_csv("./titanic/test.csv")

### セル表示
# train ## 全体
train.head(10) ## 先頭から10行目
# train.describe() ## 統計量
# train.info() ## 要約情報
# train['Age'].isnull().values.sum() ## nullの件数

### ヒストグラム
plt.hist(train['Age'].dropna(), bins=20)

### 欠損値補完
mean = np.mean(train['Age'])
train['Age'] = train['Age'].fillna(mean)
