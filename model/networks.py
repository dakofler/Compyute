class Network():
    def __init__(self) -> None:
        pass

    def validate(self, X, Y):
        pass

    def predict (self, input):
        self.layers[0].input = input


class FeedForward(Network):
    def __init__(self, layers=[]) -> None:
        super().__init__()
        self.layers = []
        if len(layers) > 0:
            for layer in layers: self.add_layer(layer)

    def add_layer(self, layer):
        if len(self.layers) == 0: layer.integrate(0)
        else:
            layer.integrate(self.layers[-1].id + 1, self.layers[-1])
        self.layers.append(layer)
    
    def __propagate(self):
        for layer in self.layers:
            layer.process()

    def train(self, train_data, epochs=100, learning_rate=0.5):
        if train_data.ndim != 3: return
        for i in range(1, epochs + 1):
            # ToDo
            pass

    def predict(self, input):
        if input.ndim != 3: return
        super().predict(input)
        self.__propagate()
        return self.layers[-1].output