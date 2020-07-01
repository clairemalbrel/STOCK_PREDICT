##########################################################################
#     Trainer
##########################################################################

# tools
import time
import warnings
import joblib
import pandas as pd
import numpy as np
from termcolor import colored

# Sklearn and machine learning tools
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, mean_squared_error
#from SentencePolarity.sentiment import Sentiment
import tornado
from SentencePolarity import sentiment

# Own libraries
from STOCK_PREDICT.data import Data

## ML FLOW
from memoized_property import memoized_property
from mlflow.tracking import MlflowClient
import mlflow

##########################################################################
warnings.filterwarnings("ignore", category=FutureWarning)

MLFLOW_URI = "http://35.210.166.253:5000"
myname="Christophe"
EXPERIMENT_NAME = "STOCK_PREDICTION"
##########################################################################


##########################################################################
#     Trainer
##########################################################################

class Trainer(object):
    ESTIMATOR = "Category"

    def __init__(self, X_train,X_test,y_train, y_test, **kwargs):

      self.pipeline = None
      self.kwargs = kwargs
      self.X_train = X_train
      self.y_train = y_train
      self.X_test = X_test
      self.y_test = y_test
      self.experiment_name = kwargs.get("experiment_name", EXPERIMENT_NAME)
      #model.set_params(**estimator_params)


    def get_estimator(self):
      estimator = self.kwargs.get("estimator", self.ESTIMATOR)
      self.mlflow_log_param("model", estimator)
      self.model = LinearDiscriminantAnalysis()
      estimator_params = self.kwargs.get("estimator_params", {})
      print(colored(self.model.__class__.__name__, "red"))
      return self.model

    def set_pipeline(self):
      self.get_estimator()

      self.pipeline = Pipeline(steps=[('scaler', StandardScaler()),
                                      ('LDA', self.model)])

    def train(self):
      tic = time.time()
      self.set_pipeline()
      self.pipeline.fit(self.X_train, self.y_train)
      self.mlflow_log_metric("train_time", int(time.time() - tic))

    def evaluate(self, X_test, y_test, show=False):
      if self.pipeline is None:
        raise ("Cannot evaluate an empty pipeline")
      y_pred = self.pipeline.predict(self.X_test)
      if show:
        res = pd.DataFrame(y_test)
        res["pred"] = y_pred
        print(colored(res.sample(5), "blue"))
      accuracy = accuracy_score(self.y_test, y_pred)
      self.mlflow_log_metric("accuracy", accuracy)
      return accuracy

    def save_model(self):
      joblib.dump(self.pipeline, 'model.joblib')
      print(colored("model.joblib saved locally", "green"))

    def compute_accuracy(self, X_test, y_test, show=False):
        if self.pipeline is None:
            raise ("Cannot evaluate an empty pipeline")
        y_pred = self.pipeline.predict(X_test)
        if show:
            res = pd.DataFrame(y_test)
            res["pred"] = y_pred
            print(colored(res.sample(5), "blue"))
        rmse = compute_rmse(y_pred, y_test)
        return round(rmse, 3)
    @memoized_property
    def mlflow_client(self):
      mlflow.set_tracking_uri(MLFLOW_URI)
      return MlflowClient()

    @memoized_property
    def mlflow_experiment_id(self):
      try:
        return self.mlflow_client.create_experiment(self.experiment_name)
      except BaseException:
        return self.mlflow_client.get_experiment_by_name(self.experiment_name).experiment_id

    @memoized_property
    def mlflow_run(self):
      return self.mlflow_client.create_run(self.mlflow_experiment_id)

    def mlflow_log_param(self, key, value):
      self.mlflow_client.log_param(self.mlflow_run.info.run_id, key, value)

    def mlflow_log_metric(self, key, value):
      self.mlflow_client.log_metric(self.mlflow_run.info.run_id, key, value)

    def log_estimator_params(self):
      reg = self.get_estimator()
      self.mlflow_log_param('estimator_name', reg.__class__.__name__)
      params = reg.get_params()
      for k, v in params.items():
        self.mlflow_log_param(k, v)

    def log_kwargs_params(self):
      if self.mlflow:
        for k, v in self.kwargs.items():
          self.mlflow_log_param(k, v)

    def log_machine_specs(self):
      cpus = multiprocessing.cpu_count()
      mem = virtual_memory()
      ram = int(mem.total / 1000000000)
      self.mlflow_log_param("ram", ram)
      self.mlflow_log_param("cpus", cpus)

##########################################################################


if __name__ == "__main__":
    # Get and clean data
    data = Data()
    df, X_train, X_test, y_train, y_test, df_train, df_test = data.clean_df()
    del df
    print("shape: {}".format(X_train.shape))

    print("size: {} Mb".format(X_train.memory_usage().sum() / 1e6))
    # Train and save model, locally and
    t = Trainer(X_train=X_train, X_test=X_test,y_train=y_train, y_test=y_test, estimator="LDA")
    print(colored("############  Training model   ############", "red"))
    t.train()
    print(colored("############  Evaluating model ############", "blue"))
    accuracy = t.evaluate(X_test=X_test, y_test=y_test)
    print(accuracy)
    print(colored("############   Saving model    ############", "green"))
    t.save_model()

