import ruamel.yaml as yaml
import qlib

config_paths_file = "pztest/exps/baselines1_dates1/config_paths.yaml"

with open(config_paths_file) as f:
        config_paths = yaml.safe_load(f)

from qlib.config import REG_CN

provider_uri="~/.qlib/qlib_data/cn_data"
region=REG_CN

task_url="mongodb://localhost:27017/"
task_db_name="rolling_db"
mongo_conf = {
    "task_url": task_url,
    "task_db_name": task_db_name,
}

qlib.init(provider_uri=provider_uri, region=region, mongo=mongo_conf)

from qlib.model.trainer import TrainerRM

experiment_name="baselines1_dates1"
task_pool="baselines1_dates1_tasks"

tasks = []
for path in config_paths.values():
    with open(path) as f:
        t = yaml.safe_load(f)
        tasks.append(t)

from qlib.workflow.task.manage import TaskManager
from qlib.workflow import R

TaskManager(task_pool=task_pool).remove()
exp = R.get_exp(experiment_name=experiment_name)
for rid in exp.list_recorders():
            exp.delete_recorder(rid)

trainer = TrainerRM(experiment_name, task_pool)
trainer.train(tasks)