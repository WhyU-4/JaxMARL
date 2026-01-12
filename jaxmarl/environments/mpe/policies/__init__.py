"""
Policy module for 3v1 pursuit-evasion environment.

This module provides modular policy implementations for both pursuers and evaders
in the 3v1 pursuit-evasion task.
"""

from .pursuer_policies import (
    PursuerPolicy,
    RandomPursuerPolicy,
    GreedyPursuerPolicy,
    CoordinatedPursuerPolicy,
    PURSUER_POLICIES,
    get_pursuer_policy,
)

from .evader_policies import (
    EvaderPolicy,
    RandomEvaderPolicy,
    BasicEvaderPolicy,
    IntermediateEvaderPolicy,
    AdvancedEvaderPolicy,
    EVADER_POLICIES,
    get_evader_policy,
)

__all__ = [
    # Pursuer policies
    "PursuerPolicy",
    "RandomPursuerPolicy",
    "GreedyPursuerPolicy",
    "CoordinatedPursuerPolicy",
    "PURSUER_POLICIES",
    "get_pursuer_policy",
    # Evader policies
    "EvaderPolicy",
    "RandomEvaderPolicy",
    "BasicEvaderPolicy",
    "IntermediateEvaderPolicy",
    "AdvancedEvaderPolicy",
    "EVADER_POLICIES",
    "get_evader_policy",
]
