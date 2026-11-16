from typing import Any, Dict, List, Tuple

from alphaapollo.core.environments.base import EnvironmentManagerBase, to_numpy

from .prompts import build_feedback_instruction


class GeometricEnvironmentManager(EnvironmentManagerBase):
    '''Environment manager shared by Circle Packing and MMD.'''

    def reset(self, kwargs) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        observations, infos = self.envs.reset(kwargs=kwargs)
        self.tasks = list(observations)
        return {
            'text': self.build_text_obs(observations, init=True),
            'image': None,
            'anchor': list(observations),
        }, infos

    def step(self, text_actions: List[str]):
        actions, valid_actions = self.projection_f(text_actions)
        observations, rewards, dones, infos = self.envs.step(actions, text_actions)
        for index, info in enumerate(infos):
            info['is_action_valid'] = to_numpy(valid_actions[index])
        return {
            'text': self.build_text_obs(observations),
            'image': None,
            'anchor': list(observations),
        }, to_numpy(rewards), to_numpy(dones), infos

    @staticmethod
    def build_text_obs(text_obs: List[str], init: bool = False) -> List[str]:
        if init:
            return list(text_obs)
        instruction = build_feedback_instruction()
        return [observation + '\n\n' + instruction for observation in text_obs]

    def _process_batch(self, batch_idx, total_batch_list, total_infos, success):
        for index in reversed(range(len(total_batch_list[batch_idx]))):
            if total_batch_list[batch_idx][index]['active_masks']:
                info = total_infos[batch_idx][index]
                value = float(info.get('valid', info.get('won', False)))
                success['success_rate'].append(value)
                source = info.get('data_source', 'unknown')
                success[f'{source}_success_rate'].append(value)
                return
