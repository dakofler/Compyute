class optimizer():
    def __init__(self) -> None:
        pass


class stochastic_gradient_descent(optimizer):
    def __init__(self, learning_rate=0.01, momentum=0.0, nesterov=False) -> None:
        super().__init__()
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.nesterov = nesterov

    def optimize(self, loss_gradient, model_layers):
        layers_reversed = model_layers.copy()
        layers_reversed.reverse()
        layers_reversed[0].loss_gradient = loss_gradient

        for layer in layers_reversed:
            g = layer.learn()
            if g is not None:
                layer.delta_weights = - self.learning_rate * g + self.momentum * layer.delta_weights

                # https://keras.io/api/optimizers/sgd/
                if not self.nesterov:
                    layer.weights = layer.weights + layer.delta_weights
                else: 
                    layer.weights = layer.weights + self.momentum * layer.delta_weights - self.learning_rate * g