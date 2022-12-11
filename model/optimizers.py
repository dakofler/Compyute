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
        layers_reversed[0].dy = loss_gradient

        for layer in layers_reversed:
            layer.learn()
            
            if layer.dw is not None:
                layer.w_change = - self.learning_rate * layer.dw + self.momentum * layer.w_change
                if not self.nesterov: layer.w = layer.w + layer.w_change
                else: layer.w = layer.w + self.momentum * layer.w_change - self.learning_rate * layer.dw # https://keras.io/api/optimizers/sgd/
            
            if layer.db is not None:
                layer.b_change = - self.learning_rate * layer.db + self.momentum * layer.b_change
                if not self.nesterov: layer.b = layer.b + layer.b_change
                else: layer.b = layer.b + self.momentum * layer.b_change - self.learning_rate * layer.db