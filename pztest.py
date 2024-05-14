from qlib.workflow.cli import workflow

if __name__ == '__main__':
    for i in range(10):
        workflow(r'examples\benchmarks\ADARNN\workflow_config_adarnn_Alpha360.yaml')