from . import butils as bu
from . import bmath as bm
import numpy as np

class NeuralNetwork:
    def __init__(self, nb_inputs=1, nb_out=1, nb_neurons_by_layer=(), copy_from=None):
        self.id = bu.getNewId()

        if copy_from is None:
            self.nb_inputs = nb_inputs
            self.nb_outputs = nb_out
            self.nb_neurons_by_layer = list(nb_neurons_by_layer)
            self.layers = []
            for nb_neurons in self.nb_neurons_by_layer:
                self.layers.append(Layer(nb_neurons,  self.nb_inputs if len(self.layers) == 0 else self.layers[-1].size_out(), network=self))
            self.layers.append(Layer(self.nb_outputs, self.nb_inputs if len(self.layers) == 0 else self.layers[-1].size_out(), network=self))
        else:
            self.nb_inputs = copy_from.nb_inputs
            self.nb_outputs = copy_from.nb_outputs
            self.nb_neurons_by_layer = list(copy_from.nb_neurons_by_layer)
            self.layers = [layer.getCopy() for layer in copy_from.layers]

    def predict(self, inputs):
        if len(inputs) != self.nb_inputs:
            print(f"Only {self.nb_inputs} inputs accepted ! : \n{inputs}")
            raise ValueError("Wrong inputs ! (size issue)")

        information = np.array(inputs)
        for layer in self.layers:
            information = layer.apply(information)

        if len(information) != self.nb_outputs:
            raise ValueError("Wrong output layer ! (size issue)")

        return information

    def NewInput(self, amount=1):
        for _ in range(amount):
            self.nb_inputs += 1
            self.layers[0].NewInput()
    def NewNeuron(self, at_layer=0, amount=1):
        if at_layer == -1:
            at_layer = len(self.layers) - 1
        for _ in range(amount):
            self.layers[at_layer].NewNeuron()
            if at_layer == len(self.layers) - 1:
                self.nb_outputs += 1
            else:
                self.layers[at_layer + 1].NewInput()
                self.nb_neurons_by_layer[at_layer] += 1

    def NewLayer(self, nb_neurons=1, layer_at=None):
        if layer_at is None : layer_at = len(self.layers) - 1 # New Layer in last just Before the out-layer)
        nb_inputs = self.layers[layer_at - 1].size_out() if layer_at - 1 >= 0 else self.nb_inputs
        self.layers.insert(layer_at, Layer(nb_neurons, nb_inputs, network=self))
        self.layers[layer_at + 1].controlInputs(nb_neurons)
        self.nb_neurons_by_layer = [self.nb_neurons_by_layer[i] for i in range(layer_at)] + [nb_neurons] + [self.nb_neurons_by_layer[i] for i in range(layer_at + 1, len(self.nb_neurons_by_layer))]

    def getSize(self):
        return sum([layer.size_out() * layer.size_out() for layer in self.layers])
    def getLayer(self, layer):
        return self.layers[layer]
    def getNeuron(self, layer, neuron):
        return self.getLayer(layer).neurons[neuron]
    def getWeight(self, layer, neuron, weight) -> float:
        return self.getNeuron(layer, neuron).weights[weight]
    def setWeight(self, weight: float, position: tuple):
        l, n, w = position
        self.getNeuron(l, n).weights[w] = weight

class Layer:
    def __init__(self, nb_neurons, nb_inputs, neurons_parameters=None, network=None):
        self.id = bu.getNewId()
        self.network = network

        self.nb_neurons = nb_neurons
        self.nb_inputs = nb_inputs
        self.neurons = [Neuron(nb_inputs, parameters=(None if neurons_parameters is None else neurons_parameters[i]), network=self.network) for i in range(nb_neurons)]

    def apply(self, inputs):
        return np.array([neuron.activate(inputs) for neuron in self.neurons])
    def size_out(self):
        return len(self.neurons)
    def size_in(self):
        return len(self.neurons[0].weights)
    def NewInput(self):
        self.nb_inputs += 1
        for neuron in self.neurons:
            neuron.AddInput()
    def NewNeuron(self):
        self.neurons.append(Neuron(self.size_in(), network=self.network))
        self.nb_neurons = len(self.neurons)
    def getCopy(self):
        return Layer(self.nb_neurons, len(self.neurons), neurons_parameters=[{'weights': neuron.weights[:], 'bias': neuron.bias} for neuron in self.neurons])
    def controlInputs(self, nb_input):
        if nb_input < self.size_in():
            for neuron in self.neurons:
                neuron.weights = np.array([neuron.weights[i] for i in range(0, nb_input)])
        elif nb_input > self.size_in():
            for i in range(nb_input - self.nb_inputs - 1):
                for neuron in self.neurons:
                    neuron.AddInput()
        self.nb_inputs = self.size_in()

class Neuron:
    def __init__(self, nb_inputs, func=bu.tanh, parameters=None, network=None):
        self.id = bu.getNewId()

        self.function = func
        self.weights = bm.normal(0, 1./np.sqrt(nb_inputs), nb_inputs) if parameters is None else parameters['weights']
        self.bias = bm.normal(0, 1.) if parameters is None else parameters['bias']
        self.network = network

    def activate(self, information):
        if len(information) != len(self.weights):
            PrintNeuralNetwork(self.network)
            raise ValueError(f"informations != nb weights : {len(information)} != {len(self.weights)}")
        return self.function(sum([self.weights[i] * information[i] for i in range(len(self.weights))]) + self.bias) # -1 -> 1

    def AddInput(self):
        self.weights = np.array(list(self.weights) + [(2 * bm.random() - 1.)/(len(self.weights)**0.5)])

def PrintNeuralNetwork(network: NeuralNetwork):
    print(f"Network Print : ({network.getSize()} conexions, {len(network.layers)} layers, {network.nb_inputs} inputs, {network.nb_outputs} outputs)")
    print(f" inputs : {network.nb_inputs}", end="")
    for i in range(len(network.layers)):
        print(f" -> ({network.layers[i].size_out()})", end="")
    print(f" -> : {network.nb_outputs} outputs")
