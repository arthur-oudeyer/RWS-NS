from .brain import NeuralNetwork
from . import bmath as bm
from tqdm import tqdm


def supervised_evaluation(prediction, input_used, target_set):
    return bm.distance(prediction, target_set[input_used])
    # for other_input in target_set:
    #     if distance > bm.distance(prediction, target_set[other_input]):
    #         return distance * 2
    # return distance

def get_score(network, inputs_set, target_set):
    score_tot = 0.
    for test_input in inputs_set:
        score_tot += supervised_evaluation(network.predict(test_input), test_input, target_set)
    return score_tot / len(inputs_set)

def fitNetwork_mutation(network: NeuralNetwork, inputs_set, target_set,  width_0=1, precision=0.01, max_iter=1000, convergence_mode='sqrt'):

    def Mutate(network: NeuralNetwork, evaluation_func, inputs_set, height=0.1, width=0.01, apply=False, structure=False):

        new_network = NeuralNetwork(copy_from=network)

        def evaluate(network_to_evaluate):
            score_tot = 0
            input_test_set = inputs_set
            for test_input in input_test_set:
                score_tot += evaluation_func(network_to_evaluate.predict(test_input), test_input, target_set)
            return score_tot / len(input_test_set)

        result_base = evaluate(network)
        new_weights = None

        for l, new_layer in enumerate(new_network.layers):
            for n, new_neuron in enumerate(new_layer.neurons):
                old_neuron = network.getNeuron(l, n)
                new_weights = old_neuron.weights[:] + height * bm.normal(0, width, len(old_neuron.weights))
                new_bias = old_neuron.bias[:] + height * bm.normal(0, width, len(old_neuron.bias))
                new_neuron.weights = new_weights
                new_neuron.bias = new_bias


        result_mutation = evaluate(new_network)

        if result_mutation < result_base and apply:
            network.layers = NeuralNetwork(copy_from=new_network).layers

        return new_weights

    width_0, width_use, width_factor = width_0, 1, 1

    prediction_score = get_score(network, inputs_set, target_set)
    learning_process = [prediction_score]

    pbar = tqdm(range(max_iter), unit='iter')
    for iter_cpt in pbar:
        pbar.set_description(f"global score: {round(learning_process[-1], 9)} | dispersion: {round(width_use, 7)} | iteration")

        if prediction_score < precision or width_factor < 1e-6:
            return learning_process

        if convergence_mode == 'sqrt':
            width_use = width_0 * ((1 - iter_cpt / max_iter) ** 0.5) * width_factor
        elif convergence_mode == 'linear':
            width_use = width_0 * (1 - iter_cpt / max_iter) * width_factor
        else:
            width_use = width_0 * width_factor

        Mutate(network, supervised_evaluation, inputs_set, height=width_use/2, width=width_use, apply=True)
        prediction_score = get_score(network, inputs_set, target_set)
        if prediction_score == learning_process[-1]:
            width_factor = max(0.01, 2 * width_0 * bm.random())
        else:
            width_factor = 1
        learning_process.append(prediction_score)

    print("[Simple Brain] Max Iteration reached or No Evolution detected in 'fitNetwork()'.")
    return learning_process


def PrefitNetwork_mutation(base_network: NeuralNetwork, inputs_set, target_set, n=-1):
    if n == -1: nb_to_generate = base_network.getSize()
    else: nb_to_generate = n

    new_network = None
    score = get_score(base_network, inputs_set, target_set)
    pbar = tqdm(range(n), unit='iter')
    for _ in pbar:
        pbar.set_description(f"initial score: {round(score, 9)} | iteration")
        new_network = NeuralNetwork(nb_inputs=base_network.nb_inputs, nb_out=base_network.nb_outputs, nb_neurons_by_layer=base_network.nb_neurons_by_layer)
        new_score = get_score(new_network, inputs_set, target_set)
        if new_score < score:
            base_network = new_network
            score = new_score

    return base_network
