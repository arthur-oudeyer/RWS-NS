"""
saver.py
========
Save and load NeuralNetwork controllers to/from disk.

Weights and biases are serialised explicitly (not the NeuralNetwork object
itself) so saves remain valid even if the NeuralNetwork class changes.

Save format v2.0 — each robot is stored as a self-contained (network, morphology)
pair so heterogeneous populations are represented correctly.

Usage:
    from saver import save_controller, load_controller
    from simplebrain_loc.brain import NeuralNetwork

    # Save a list of networks with metadata
    save_controller(
        networks     = controllers,          # list[NeuralNetwork]
        name         = "gen_42",
        context      = {"generation": 42, "best_score": 3.7},
        morphologies = robot_morphologies,   # list[RobotMorphology] — optional
    )

    # Load back
    payload      = load_controller("gen_42")
    networks     = payload["networks"]      # list[NeuralNetwork], ready to use
    morphologies = payload["morphologies"]  # list[RobotMorphology]
    context      = payload["context"]       # score, robot_index, …
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
    networks:     list[NeuralNetwork] | NeuralNetwork,
    name:         str,
    context:      dict = None,
    morphologies: list = None,
) -> str:
    """
    Serialise one or several NeuralNetworks to  saves/<name>.pkl  (format v2.0).

    Each robot is stored as a self-contained (network, morphology) pair so
    heterogeneous populations — where every robot may have a different body
    and thus different input/output dimensions — are represented correctly.

    Parameters
    ----------
    networks     : single NeuralNetwork or list of NeuralNetworks
    name         : filename stem (no extension), e.g. "gen_42"
    context      : any extra metadata to store (generation, score, config…)
    morphologies : list[RobotMorphology] aligned with networks, or None

    Returns
    -------
    Full path of the saved file.
    """
    if isinstance(networks, NeuralNetwork):
        networks = [networks]

    if morphologies is not None and not isinstance(morphologies, list):
        morphologies = [morphologies] * len(networks)

    if morphologies is not None:
        from morphology import morphology_to_dict
        morph_dicts = [morphology_to_dict(m) for m in morphologies]
    else:
        morph_dicts = [None] * len(networks)

    payload = {
        "version":  "2.0",
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "context":  context or {},
        "robots": [
            {
                "network":    _network_to_dict(net),
                "morphology": morph_dicts[i],
            }
            for i, net in enumerate(networks)
        ],
    }

    os.makedirs(SAVES_DIR, exist_ok=True)
    path = os.path.join(SAVES_DIR, f"{name}.pkl")
    with open(path, "wb") as f:
        pickle.dump(payload, f)

    print(f"[saver] Saved {len(networks)} robot(s) → {name}.pkl")
    return path


def load_controller(name: str) -> dict:
    """
    Load a saved controller from  saves/<name>.pkl

    Supports both v2.0 (per-robot entries) and v1.0 (parallel lists) files.

    Returns a dict with keys:
      "context"       — metadata stored at save time
      "saved_at"      — ISO timestamp string
      "networks"      — list[NeuralNetwork], ready to use
      "morphologies"  — list[RobotMorphology] (empty list if not stored)

    Parameters
    ----------
    name : filename stem (no extension), e.g. "gen_42"
    """
    path = os.path.join(SAVES_DIR, f"{name}.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"[saver] No save found at: {path}")

    with open(path, "rb") as f:
        payload = pickle.load(f)

    version = payload.get("version", "1.0")

    if version == "2.0":
        network_dicts  = [r["network"]    for r in payload["robots"]]
        morph_dicts    = [r["morphology"] for r in payload["robots"]]
    else:
        # v1.0 backward-compat: parallel lists
        network_dicts = payload["networks"]
        morph_dicts   = payload.get("morphologies") or [None] * len(network_dicts)

    networks = [_dict_to_network(d) for d in network_dicts]

    if any(m is not None for m in morph_dicts):
        from morphology import dict_to_morphology
        morphologies = [dict_to_morphology(m) if m is not None else None for m in morph_dicts]
    else:
        morphologies = []

    print(f"[saver] Loaded {len(networks)} robot(s) from {name}.pkl  (v{version}, saved {payload['saved_at']})")

    return {
        "context":      payload["context"],
        "saved_at":     payload["saved_at"],
        "networks":     networks,
        "morphologies": morphologies,
    }


def list_saves() -> list[str]:
    """Return all save names available in the saves/ directory."""
    if not os.path.isdir(SAVES_DIR):
        return []
    return [f[:-4] for f in os.listdir(SAVES_DIR) if f.endswith(".pkl")]


def clear_save(name: str) -> bool:
    """
    Delete the save file  saves/<name>.pkl  if it exists.

    Returns True if the file was deleted, False if it did not exist.
    """
    path = os.path.join(SAVES_DIR, f"{name}.pkl")
    if os.path.exists(path):
        os.remove(path)
        print(f"[saver] Cleared save: {name}.pkl")
        return True
    print(f"[saver] Nothing to clear — '{name}.pkl' does not exist.")
    return False