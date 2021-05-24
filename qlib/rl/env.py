# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Union

from .interpreter import StateInterpreter, ActionInterpreter
from ..contrib.backtest.executor import BaseExecutor
from ..utils import init_instance_by_config


class BaseRLEnv:
    """Base environment for reinforcement learning"""

    def reset(self, **kwargs):
        raise NotImplementedError("reset is not implemented!")

    def step(self, action):
        """
        step method of rl env
        Parameters
        ----------
        action :
            action from rl policy

        Returns
        -------
        env state to rl policy
        """
        raise NotImplementedError("step is not implemented!")


class QlibRLEnv:
    """qlib-based RL env"""

    def __init__(
        self,
        executor: BaseExecutor,
    ):
        """
        Parameters
        ----------
        executor : BaseExecutor
            qlib multi-level/single-level executor, which can be regarded as gamecore in RL
        """
        self.executor = executor

    def reset(self, **kwargs):
        self.executor.reset(**kwargs)


class QlibIntRLEnv(QlibRLEnv):
    """(Qlib)-based RL (Env) with (Interpreter)"""

    def __init__(
        self,
        executor: BaseExecutor,
        state_interpreter: Union[dict, StateInterpreter],
        action_interpreter: Union[dict, ActionInterpreter],
    ):
        """

        Parameters
        ----------
        state_interpreter : Union[dict, StateInterpreter]
            interpretor that interprets the qlib execute result into rl env state.

        action_interpreter : Union[dict, ActionInterpreter]
            interpretor that interprets the rl agent action into qlib order list
        """
        super(QlibIntRLEnv, self).__init__(executor=executor)
        self.state_interpreter = init_instance_by_config(state_interpreter)
        self.action_interpreter = init_instance_by_config(action_interpreter)

    def step(self, action):
        """
        step method of rl env, it run as following step:
            - Use `action_interpreter.interpret` method to interpret the agent action into order list
            - Execute the order list with qlib executor, and get the executed result
            - Use `state_interpreter.interpret` method to interpret the executed result into env state

        Parameters
        ----------
        action :
            action from rl policy

        Returns
        -------
        env state to rl policy
        """
        _interpret_action = self.action_interpreter.interpret(action=action)
        _execute_result = self.executor.execute(_interpret_action)
        _interpret_state = self.state_interpreter.interpret(execute_result=_execute_result)
        return _interpret_state