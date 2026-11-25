"""Attack package: objectives, the PGD engine, stealth, and site search."""

from .objectives import JailbreakObjective, TargetSequenceObjective
from .pgd import LatentPGDAttack
from .stealth import StealthRegularizer, fit_centroid, project_to_norm_ball
from .search import SiteSpace, search_site
from .grpo import GRPOReachAttack, Reward
from .sae import SAESteerer

__all__ = [
    "JailbreakObjective",
    "TargetSequenceObjective",
    "LatentPGDAttack",
    "StealthRegularizer",
    "fit_centroid",
    "project_to_norm_ball",
    "SiteSpace",
    "search_site",
    "GRPOReachAttack",
    "Reward",
    "SAESteerer",
]
