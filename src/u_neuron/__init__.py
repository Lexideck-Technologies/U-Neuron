"""U-Neuron library — Foliated U-space complex-valued neural architecture.

Public API:
    Data structures:  UTensor
    Layers:           ULinear, UEmission, UModel
    Functions:        u_norm, u_distance, u_emit
    Activations:      CReLU, modReLU
    Regularization:   LandauerRegularizer
"""

from u_neuron.activations import CReLU, modReLU
from u_neuron.emission import UEmission, u_emit
from u_neuron.model import UModel
from u_neuron.norm import u_distance, u_norm
from u_neuron.regularization import LandauerRegularizer
from u_neuron.ulinear import ULinear
from u_neuron.utensor import UTensor

__version__ = "0.1.0"

__all__ = [
    "UTensor",
    "u_norm",
    "u_distance",
    "ULinear",
    "CReLU",
    "modReLU",
    "u_emit",
    "UEmission",
    "LandauerRegularizer",
    "UModel",
]
