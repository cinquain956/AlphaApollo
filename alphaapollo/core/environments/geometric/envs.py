import asyncio
import concurrent.futures
from typing import Any, Dict, List

import gymnasium as gym

from .env import GeometricTextEnv
from .spec import get_problem_spec


class GeometricMultiProcessEnv(gym.Env):
    def __init__(
        self,
        seed: int = 0,
        env_num: int = 1,
        group_n: int = 1,
        is_train: bool = True,
        env_config: Any = None,
    ) -> None:
        super().__init__()
        del seed, is_train
        self.batch_size = int(env_num) * int(group_n)
        if self.batch_size < 1:
            raise ValueError('batch size must be positive')
        self.problem_type = str(env_config.env_name).lower()
        get_problem_spec(self.problem_type)
        self.max_steps = int(getattr(env_config, 'max_steps', 1))
        self.envs = [
            GeometricTextEnv(self.problem_type, env_config)
            for _ in range(self.batch_size)
        ]
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=min(self.batch_size, 256)
        )
        self._loop = asyncio.new_event_loop()
        self._closed = False

    def _sync_reset(self, env: GeometricTextEnv, kwargs: Dict[str, Any]):
        extras = dict(kwargs)
        extras.setdefault('max_steps', self.max_steps)
        env.reset(extras)
        return env.problem_statement(), {
            'data_source': extras.get('data_source', self.problem_type),
            'instance_params': dict(env.instance_params),
        }

    @staticmethod
    def _sync_step(env: GeometricTextEnv, action: str, text_action: str):
        output = env.step(action, text_action)
        observations = output['observations']
        observation = '' if not observations else observations[0]['content'].strip()
        metadata = output.get('metadata', {})
        info = {
            **metadata,
            'postprocessed_action': output.get('postprocessed_action'),
            'won': bool(metadata.get('valid', False)),
        }
        return observation, output['reward'], output['done'], info

    def reset(self, kwargs: List[Dict[str, Any]]):
        if len(kwargs) > self.batch_size:
            raise ValueError(
                f'got {len(kwargs)} reset items for batch size {self.batch_size}'
            )
        valid_count = len(kwargs)
        padded = list(kwargs) + [{} for _ in range(self.batch_size - valid_count)]
        tasks = [
            self._loop.run_in_executor(self._executor, self._sync_reset, env, item)
            for env, item in zip(self.envs, padded)
        ]
        results = self._loop.run_until_complete(asyncio.gather(*tasks))
        observations, infos = map(list, zip(*results))
        return observations[:valid_count], infos[:valid_count]

    def step(self, actions: List[str], text_actions: List[str]):
        if len(actions) > self.batch_size:
            raise ValueError(
                f'got {len(actions)} actions for batch size {self.batch_size}'
            )
        valid_count = len(actions)
        padded_actions = list(actions) + [''] * (self.batch_size - valid_count)
        padded_text = list(text_actions) + [''] * (self.batch_size - len(text_actions))
        tasks = [
            self._loop.run_in_executor(
                self._executor,
                self._sync_step,
                env,
                action,
                text_action,
            )
            for env, action, text_action in zip(self.envs, padded_actions, padded_text)
        ]
        results = self._loop.run_until_complete(asyncio.gather(*tasks))
        observations, rewards, dones, infos = map(list, zip(*results))
        return (
            observations[:valid_count],
            rewards[:valid_count],
            dones[:valid_count],
            infos[:valid_count],
        )

    def close(self) -> None:
        if self._closed:
            return
        for env in self.envs:
            env.close()
        self._executor.shutdown(wait=True)
        if not self._loop.is_closed():
            self._loop.close()
        self._closed = True


def build_geometric_envs(
    seed: int = 0,
    env_num: int = 1,
    group_n: int = 1,
    is_train: bool = True,
    env_config: Any = None,
):
    return GeometricMultiProcessEnv(
        seed=seed,
        env_num=env_num,
        group_n=group_n,
        is_train=is_train,
        env_config=env_config,
    )

