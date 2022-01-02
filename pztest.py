# from qlib.workflow.cli import workflow

import qlib.workflow.cli as cli

if __name__ == "__main__":

    cli.workflow('examples/benchmarks/LSTM/workflow_config_lstm_Alpha158_batch800_job8.yaml', experiment_name='pztest3')
