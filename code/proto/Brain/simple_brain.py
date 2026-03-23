from simplebrain_loc.brain import NeuralNetwork
from simplebrain_loc.bmath import normal
from saver import load_controller, list_saves
import numpy as np

# ---------- Config ------------ #
NB_INPUT_CLOCKS = 3
NB_INPUTS = NB_INPUT_CLOCKS + 6 # + 10
NB_OUTPUTS = 3
NB_NEURONS_BY_LAYER = (2 * NB_INPUTS, 2 * NB_INPUTS, NB_INPUTS)
PREDICTION_FACTOR = -2 # tune sensibility
# ------------------------------ #

controllers = []

def _fresh() -> NeuralNetwork:
    return NeuralNetwork(NB_INPUTS, NB_OUTPUTS, NB_NEURONS_BY_LAYER)

def init_simplebrain_controllers(N, init_config=None):
    """
    Initialise the N controllers according to init_config (from sim_config.CONTROLLER_INIT).

    init_config formats:
      None              → all fresh
      "latest"          → load all from the latest best_* save
      "last_sim"        → load all from the last_sim save
      {"source": ...,   → load only `indices` from source, rest fresh
       "indices": [...] or "all"}
    """
    global controllers
    controllers.clear()

    if init_config is None:
        controllers.extend(_fresh() for _ in range(N))
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
        print(f"[init] Save '{source}' not found — starting fresh.")
        controllers.extend(_fresh() for _ in range(N))
        return

    # --- build controller list ---
    if load_indices == "all":
        load_indices = list(range(min(N, len(saved_networks))))

    load_set = set(load_indices)
    for i in range(N):
        if i in load_set and i < len(saved_networks):
            controllers.append(NeuralNetwork(copy_from=saved_networks[i]))
        else:
            controllers.append(_fresh())

    loaded = len(load_set & set(range(len(saved_networks))))
    print(f"[init] Loaded {loaded} controller(s) from '{source}', {N - loaded} fresh.")

def get_input_clocks(time):
    return np.sin(time * 15 / 3.142), np.sin(time * 5 / 3.142), np.sin(time * 1 / 3.142)

def get_simplebrain_controller():

    def controller(robot_index, current_time, N_HIP, sensors):
        global controllers
        dp = PREDICTION_FACTOR * controllers[robot_index].predict((*get_input_clocks(current_time), *sensors.getHipsData()))
        #print(f"robot n°{robot_index} -> {sensors.hip_angles}, {dp}, {sensors.hip_angles + dp}")
        return sensors.hip_angles + dp

    return controller

def Mutate(network: NeuralNetwork, amplitude=0.1, variation=0.01):
    new_network = NeuralNetwork(copy_from=network)

    for l, new_layer in enumerate(new_network.layers):
        for n, new_neuron in enumerate(new_layer.neurons):
            old_neuron = network.getNeuron(l, n)
            new_neuron.weights = old_neuron.weights[:] + amplitude * normal(0, variation, len(old_neuron.weights))
            new_neuron.bias = old_neuron.bias + amplitude * normal(0, variation)

    return new_network




