from azureml.core import Run
import pandas as pd
import os

# 実験実行コンテキストの取得
run = Run.get_context()

# 糖尿病データセットの読み込み
data = pd.read_csv('diabetes.csv')

# 行を数え、結果を記録
row_count = (len(data))
run.log('observations', row_count)
print('Analyzing {} rows of data'.format(row_count))

# ラベルの数を数えて記録
diabetic_counts = data['Diabetic'].value_counts()
print(diabetic_counts)
for k, v in diabetic_counts.items():
    run.log('Label:' + str(k), v)
      
# データのサンプルをoutputフォルダに保存(自動的にアップロードされる
os.makedirs('outputs', exist_ok=True)
data.sample(100).to_csv("outputs/sample.csv", index=False, header=True)

# 実行の終了
run.complete()
