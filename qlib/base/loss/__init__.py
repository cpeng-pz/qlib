from qlib.base.loss.adv_loss import adv
from qlib.base.loss.coral import CORAL
from qlib.base.loss.cos import cosine
from qlib.base.loss.kl_js import kl_div, js
from qlib.base.loss.mmd import MMD_loss
from qlib.base.loss.mutual_info import Mine
from qlib.base.loss.pair_dist import pairwise_dist

__all__ = [
    'adv',
    'CORAL',
    'cosine',
    'kl_div',
    'js'
    'MMD_loss',
    'Mine',
    'pairwise_dist'
]