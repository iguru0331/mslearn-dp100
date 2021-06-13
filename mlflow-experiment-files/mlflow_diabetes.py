from azureml.core import Run
import pandas as pd
import mlflow


# MLflow実験の開始
with mlflow.start_run():
       
    # データの読み込み
    data = pd.read_csv('diabetes.csv')

    # 行数をカウントし、結果をロギング
    row_count = (len(data))
    print('observations:', row_count)
    mlflow.log_metric('observations', row_count)
