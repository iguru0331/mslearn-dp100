# Import libraries
import os
import argparse
from azureml.core import Dataset, Run
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import glob

# スクリプト引数を取得　※正規化率とファイルデータセットのマウントポイント
parser = argparse.ArgumentParser()
parser.add_argument('--regularization', type=float, dest='reg_rate', default=0.01, help='regularization rate')
parser.add_argument('--input-data', type=str, dest='dataset_folder', help='data mount point')
args = parser.parse_args()

# 正則化ハイパーパラメータの設定(スクリプトの引数として渡される)
reg = args.reg_rate

# 実験の実行コンテキストを取得
run = Run.get_context()

# 糖尿病データの読み取り
print("Loading Data...")
data_path = run.input_datasets['training_files'] # 入力からトレーニングデータのパスを取得
# ※ハードコードされたフレンドなり-な名前に頼りたくない場合、args.dataset_folderを使うこともできる

# ファイルの読み込み
all_files = glob.glob(data_path + "/*.csv")
diabetes = pd.concat((pd.read_csv(f) for f in all_files), sort=False)

# 特徴量とラベルの分離
X, y = diabetes[['Pregnancies','PlasmaGlucose','DiastolicBloodPressure','TricepsThickness','SerumInsulin','BMI','DiabetesPedigree','Age']].values, diabetes['Diabetic'].values

# 訓練データとテストデータの分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

# ロジスティック回帰モデルの訓練
print('Training a logistic regression model with regularization rate of', reg)
run.log('Regularization Rate',  np.float(reg))
model = LogisticRegression(C=1/reg, solver="liblinear").fit(X_train, y_train)

# 精度計算
y_hat = model.predict(X_test)
acc = np.average(y_hat == y_test)
print('Accuracy:', acc)
run.log('Accuracy', np.float(acc))

# AUC計算
y_scores = model.predict_proba(X_test)
auc = roc_auc_score(y_test,y_scores[:,1])
print('AUC: ' + str(auc))
run.log('AUC', np.float(auc))

os.makedirs('outputs', exist_ok=True)
# ※ハードコードされたフレンドリーな名前に頼りたくない場合は、args.dataset_folderを使うこともできる
joblib.dump(value=model, filename='outputs/diabetes_model.pkl')

run.complete()
