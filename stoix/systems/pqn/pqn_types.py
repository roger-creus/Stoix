from typing import Dict

import chex
from typing_extensions import NamedTuple

from stoix.base_types import Action, Done, Truncated, Value

class PQNTransition(NamedTuple):
    """Transition tuple for PQN."""

    done: Done
    truncated: Truncated
    action: Action
    value: Value
    reward: chex.Array
    obs: chex.Array
    info: Dict


