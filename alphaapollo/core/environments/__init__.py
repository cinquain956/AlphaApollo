# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
# Copyright 2026 TMLR Group
# Licensed under the Apache License, Version 2.0.

__all__ = ['EnvironmentManagerBase', 'make_envs']


def __getattr__(name):
    if name in __all__:
        from alphaapollo.core.environments.env_manager import (
            EnvironmentManagerBase,
            make_envs,
        )

        return {
            'EnvironmentManagerBase': EnvironmentManagerBase,
            'make_envs': make_envs,
        }[name]
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
