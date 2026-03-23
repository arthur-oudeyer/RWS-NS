"""
saver.py
========
Save and load NeuralNetwork controllers to/from disk.

Weights and biases are serialised explicitly (not the NeuralNetwork object
itself) so saves remain valid even if the NeuralNetwork class changes.

Usage:
    from saver import save_controller, load_controller
    from simplebrain_loc.brain import NeuralNetwork

    # Save a list of networks with metadata
    save_controller(
        networks   = controllers,          # list[NeuralNetwork]
        name       = "gen_42",
        context    = {"generation": 42, "best_score": 3.7}
    )

    # Load back
    payload = load_controller("gen_42")
    architecture = payload["architecture"]
    context      = payload["context"]
    networks     = payload["networks"]    # list[NeuralNetwork], ready to use
"""

import os
import pickle
from datetime import datetime

import numpy as np

import sys
sys.path.insert(0, os.path.dirname(__file__))
from simplebrain_loc.brain import NeuralNetwork

SAVES_DIR = os.path.join(os.path.dirname(__file__), "saves")


# ---------------------------------------------------------------------------
# Internal helpers — convert NeuralNetwork ↔ plain serialisable dict
# ---------------------------------------------------------------------------
def _network_to_dict(network: NeuralNetwork) -> dict:
    """Extract all weights/biases into a plain dict (no custom objects)."""
    return {
        "nb_inputs":           network.nb_inputs,
        "nb_outputs":          network.nb_outputs,
        "nb_neurons_by_layer": list(network.nb_neurons_by_layer),
        "layers": [
            [
                {
                    "weights": neuron.weights.tolist(),
                    "bias":    float(neuron.bias),
                }
                for neuron in layer.neurons
            ]
            for layer in network.layers
        ],
    }


def _dict_to_network(d: dict) -> NeuralNetwork:
    """Reconstruct a NeuralNetwork from a serialised dict."""
    network = NeuralNetwork(
        nb_inputs          = d["nb_inputs"],
        nb_out             = d["nb_outputs"],
        nb_neurons_by_layer= d["nb_neurons_by_layer"],
    )
    for layer, layer_data in zip(network.layers, d["layers"]):
        for neuron, neuron_data in zip(layer.neurons, layer_data):
            neuron.weights = np.array(neuron_data["weights"])
            neuron.bias    = neuron_data["bias"]
    return network


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def save_controller(
    networks:  list[NeuralNetwork] | NeuralNetwork,
    name:      str,
    context:   dict = None,
) -> str:
    """
    Serialise one or several NeuralNetworks to  saves/<name>.pkl

    Parameters
    ----------
    networks  : single NeuralNetwork or list of NeuralNetworks
    name      : filename stem (no extension), e.g. "gen_42"
    context   : any extra metadata to store (generation, score, config…)

    Returns
    -------
    Full path of the saved file.
    """
    if isinstance(networks, NeuralNetwork):
        networks = [networks]

    first = networks[0]

    payload = {
        "version":      "1.0",
        "saved_at":     datetime.now().isoformat(timespec="seconds"),
        "context":      context or {},
        "architecture": {
            "nb_inputs":           first.nb_inputs,
            "nb_outputs":          first.nb_outputs,
            "nb_neurons_by_layer": list(first.nb_neurons_by_layer),
            "n_networks":          len(networks),
        },
        "networks": [_network_to_dict(n) for n in networks],
    }

    os.makedirs(SAVES_DIR, exist_ok=True)
    path = os.path.join(SAVES_DIR, f"{name}.pkl")
    with open(path, "wb") as f:
        pickle.dump(payload, f)

    print(f"[saver] Saved {len(networks)} network(s) → {path}")
    return path


def load_controller(name: str) -> dict:
    """
    Load a saved controller from  saves/<name>.pkl

    Returns a dict with keys:
      "architecture"  — nb_inputs, nb_outputs, nb_neurons_by_layer, n_networks
      "context"       — metadata stored at save time
      "saved_at"      — ISO timestamp string
      "networks"      — list[NeuralNetwork], ready to use

    Parameters
    ----------
    name : filename stem (no extension), e.g. "gen_42"
    """
    path = os.path.join(SAVES_DIR, f"{name}.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"[saver] No save found at: {path}")

    with open(path, "rb") as f:
        payload = pickle.load(f)

    networks = [_dict_to_network(d) for d in payload["networks"]]

    print(f"[saver] Loaded {len(networks)} network(s) from {path}  (saved {payload['saved_at']})")

    return {
        "architecture": payload["architecture"],
        "context":      payload["context"],
        "saved_at":     payload["saved_at"],
        "networks":     networks,
    }


def list_saves() -> list[str]:
    """Return all save names available in the saves/ directory."""
    if not os.path.isdir(SAVES_DIR):
        return []
    return [f[:-4] for f in os.listdir(SAVES_DIR) if f.endswith(".pkl")]