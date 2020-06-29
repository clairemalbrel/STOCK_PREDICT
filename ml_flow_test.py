import mlflow
from  mlflow.tracking import MlflowClient

EXPERIMENT_NAME = "test_experiment"
# Indicate mlflow to log to remote server
mlflow.set_tracking_uri("http://35.210.166.253:5000")
client = MlflowClient()
try:
  experiment_id = client.create_experiment(EXPERIMENT_NAME)
except BaseException:
  experiment_id = client.get_experiment_by_name(EXPERIMENT_NAME).experiment_id

yourname='Christophe'

if not yourname:
  print("please define your name, il will be used as a parameter to log")

for model in ["Category", "LDA"]:
  run = client.create_run(experiment_id)
  client.log_metric(run.info.run_id, "Accuracy", 4.5)
  client.log_param(run.info.run_id, "model", model)
  client.log_param(run.info.run_id, "student_name", yourname)
