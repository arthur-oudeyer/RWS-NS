from simplebrain_loc.brain import NeuralNetwork, PrintNeuralNetwork
from simplebrain_loc.bmath import normal
from saver import load_controller, list_saves
import numpy as np
import random
from morphology import RobotMorphology, QUADRIPOD, pad_morphologies

# ---------- Config ------------ #
NB_INPUT_CLOCKS = 3
PREDICTION_FACTOR = -5 # tune sensibility
# ------------------------------ #

def _nb_inputs(morph: RobotMorphology) -> int:
    return NB_INPUT_CLOCKS + morph.n_sensor_inputs

def _nb_outputs(morph: RobotMorphology) -> int:
    return morph.n_outputs

def _nb_neurons(morph: RobotMorphology) -> tuple:
    #nb_in = _nb_inputs(morph)
    return (20, 20, 10) # To change ?

controllers = []

def _fresh(morph: RobotMorphology = QUADRIPOD) -> NeuralNetwork:
    return NeuralNetwork(_nb_inputs(morph), _nb_outputs(morph), _nb_neurons(morph))

def init_simplebrain_controllers(N, init_config=None, morphologies=None):
    """
    Initialise the N controllers according to init_config (from sim_config.CONTROLLER_INIT).

    init_config formats:
      None              → all fresh
      "last_best"          → load all from the latest best_* save
      "last_sim"        → load all from the last_sim save
      {"source": ...,   → load only `indices` from source, rest fresh
       "indices": [...] or "all"}
    """
    global controllers
    controllers.clear()

    _morphologies = pad_morphologies(N, morphologies or QUADRIPOD)

    if init_config is None:
        print(f"[init morphology] starting fresh controllers.")
        controllers.extend(_fresh(_morphologies[i]) for i in range(N))
        return

    # --- parse config --- (last_sim / last_best)
    if isinstance(init_config, str):
        source       = init_config
        load_indices = "all"
    else:
        source       = init_config["source"]
        load_indices = init_config.get("indices", "all")

    # --- load the save ---
    try:
        payload        = load_controller(source)
        saved_networks = payload["networks"]
    except FileNotFoundError:
        print(f"[init brain] Save '{source}' not found — starting fresh.")
        controllers.extend(_fresh(_morphologies[i]) for i in range(N))
        return

    # --- build controller list ---
    if load_indices == "mutation":
        # All robots pick a random elite as seed and are passed through Mutate() so
        # their dimensions are always reconciled with their assigned morphology.
        # Robot 0 uses amplitude=0 (weights unchanged, only dimensions adapted if needed).
        amp = init_config.get("amplitude") or 0.1
        var = init_config.get("variation") or 0.01
        for i in range(N):
            source_net = random.choice(saved_networks)
            controllers.append(Mutate(source_net, morph=_morphologies[i],
                                      amplitude=0.0 if i == 0 else amp,
                                      variation=var))
        print(f"[init brain] Loaded {len(saved_networks)} elite(s) from '{source}', filled {N} robot(s).")
        return

    if load_indices == "all":
        load_indices = list(range(min(N, len(saved_networks))))

    load_set = set(load_indices)
    for i in range(N):
        morph = _morphologies[i]
        if i in load_set and i < len(saved_networks):
            net = NeuralNetwork(copy_from=saved_networks[i])
            # Check dimensions match this robot's morphology
            if net.nb_inputs != _nb_inputs(morph) or net.nb_outputs != _nb_outputs(morph):
                print(f"[init brain] R{i}: saved dimensions ({net.nb_inputs}in, {net.nb_outputs}out) "
                      f"don't match morphology {morph.name} "
                      f"({_nb_inputs(morph)}in, {_nb_outputs(morph)}out) — creating fresh.")
                controllers.append(_fresh(morph))
            else:
                controllers.append(net)
        else:
            controllers.append(_fresh(morph))

    loaded = len(load_set & set(range(len(saved_networks))))
    print(f"[init brain] Loaded {loaded} controller(s) from '{source}', {N - loaded} fresh.")

def get_input_clocks(time):
    return np.sin(time * 15 / 3.142), np.sin(time * 5 / 3.142), np.sin(time * 1 / 3.142)

def get_simplebrain_controller():

    def controller(robot_index, current_time, N_HIP, sensors):
        global controllers
        dp = PREDICTION_FACTOR * controllers[robot_index].predict((*get_input_clocks(current_time), *sensors.getHipsData()))
        #print(f"robot n°{robot_index} -> {sensors.hip_angles}, {dp}, {sensors.hip_angles + dp}")
        return sensors.hip_angles + dp

    return controller

def Mutate(network: NeuralNetwork, morph: RobotMorphology, amplitude=0.1, variation=0.01):
    amplitude = amplitude if amplitude is not None else 0.1
    variation = variation if variation is not None else 0.01
    new_network = NeuralNetwork(copy_from=network)

    for l, new_layer in enumerate(new_network.layers):
        for n, new_neuron in enumerate(new_layer.neurons):
            old_neuron = network.getNeuron(l, n)
            new_neuron.weights = old_neuron.weights[:] + amplitude * normal(0, variation, len(old_neuron.weights))
            new_neuron.bias = old_neuron.bias + amplitude * normal(0, variation)

    while new_network.nb_inputs < _nb_inputs(morph):
        new_network.NewInput()
    while new_network.nb_inputs > NB_INPUT_CLOCKS + morph.n_sensor_inputs:
        new_network.RemoveInput()

    while new_network.nb_outputs < _nb_outputs(morph):
        new_network.NewOutput()
    while new_network.nb_outputs > _nb_outputs(morph):
        new_network.RemoveOutput(rnd=True)

    return new_network




