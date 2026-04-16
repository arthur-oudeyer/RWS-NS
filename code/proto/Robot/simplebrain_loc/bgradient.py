from .brain import NeuralNetwork
from . import bmath as bm
from tqdm import tqdm

def GradientSimplified(network: NeuralNetwork, evaluation_func, inputs_set, alpha=0.01, apply=False):

    def evaluate(network_to_evaluate):
        score_tot = 0
        input_test_set = inputs_set #bm.ShuffledOf(inputs_set)[:len(inputs_set)//4]
        for test_input in input_test_set:
            score_tot += evaluation_func(network_to_evaluate.predict(test_input), test_input)
        return score_tot / len(input_test_set)

    gradients = [[[0. for _ in neuron.weights] for neuron in layer.neurons] for layer in network.layers]

    result_base = evaluate(network)

    for l, layer in enumerate(network.layers):
        for n, neuron in enumerate(layer.neurons):
            for w, weight in enumerate(neuron.weights):
                old_weight_tmp = network.getWeight(l, n, w)

                network.setWeight(old_weight_tmp + alpha, (l, n, w))
                result_1 = evaluate(network)

                network.setWeight(old_weight_tmp - alpha, (l, n, w))
                result_2 = evaluate(network)

                if result_1 < result_base and result_1 < result_2:
                    gradients[l][n][w] = old_weight_tmp + alpha
                    network.setWeight(old_weight_tmp + alpha, (l, n, w))
                elif result_2 < result_base:
                    gradients[l][n][w] = old_weight_tmp - alpha
                    network.setWeight(old_weight_tmp - alpha, (l, n, w))
                else:
                    gradients[l][n][w] = old_weight_tmp
                    network.setWeight(old_weight_tmp, (l, n, w))

                if not apply:
                    network.setWeight(old_weight_tmp, (l, n, w))

    return gradients

def fitNetwork_simple(network: NeuralNetwork, inputs_set, target_set, precision=0.01, max_iter=1000, alpha=0.001, alpha_mode='linear'):

    target_to_fit = {inputs_set[i]: target_set[i] for i in range(len(inputs_set))}
    def Supervised_Evaluation(prediction, input_used):
        return (10 * bm.distance(prediction, target_to_fit[input_used]))**2

    def getScore():
        score_tot = 0.
        for test_input in inputs_set:
            score_tot += Supervised_Evaluation(network.predict(test_input), test_input)
        return score_tot / len(inputs_set)

    prediction_score = getScore()
    alpha_0, alpha_use, alpha_factor = alpha, alpha, 1

    learning_process = [prediction_score]

    pbar = tqdm(range(max_iter), unit='iter')
    for iter_cpt in pbar:
        pbar.set_description(f"global score: {round(learning_process[-1], 9)} | alpha: {round(alpha_use, 7)} | iteration")
        if prediction_score < precision or alpha_factor < 1e-6:
            return learning_process

        if alpha_mode == 'linear':
            alpha_use = alpha * ((1 - iter_cpt / max_iter) ** 0.5) * alpha_factor
        else:
            alpha_use = alpha_0 * alpha_factor

        GradientSimplified(network, Supervised_Evaluation, inputs_set, alpha_use, apply=True)
        prediction_score = getScore()
        if prediction_score == learning_process[-1]:
            alpha_factor = max(0.1, 2 * alpha_0 * bm.random())
        learning_process.append(prediction_score)

    print("[Simple Brain] Max Iteration reached or No Evolution detected in 'fitNetwork()'.")
    return learning_process
