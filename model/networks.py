class Network():
    def __init__(self) -> None:
        pass

    def validate(self, X, Y):
        pass

    def predict (self, input):
        if input.shape != self.layers[0].input_shape:
            print('Input shape does not match configuration.')
            return None
        self.layers[0].input = input


class FeedForward(Network):
    def __init__(self, layers=[]) -> None:
        super().__init__()
        self.layers = []
        if len(layers) > 0:
            for layer in layers: self.add_layer(layer)

    def add_layer(self, layer):
        if len(self.layers) == 0: layer.integrate(0, None)
        else:
            layer.integrate(self.layers[-1].id + 1, self.layers[-1])
        self.layers.append(layer)
    
    def __propagate(self):
        for layer in self.layers:
            layer.process()

    def train(self, X, Y):
        pass

    def predict(self, input):
        super().predict(input)
        self.__propagate()
        return self.layers[-1].output  


# class Network:

# class Feed_Forward(Network):
#     def train(self,
#         train_df_x: pandas.DataFrame,
#         train_df_y: pandas.DataFrame,
#         mode: str = 'online',
#         epochs: int = 100,
#         max_error = 0.0,
#         adaptive_learning_rate: bool = False,
#         min_learning_rate = 0.0,
#         default_learning_rate = 0.5,
#         momentum_factor = 0.0,
#         flatspot_elim_value = 0.1,
#         weight_decay_factor = 0.0,
#         rprop = False,
#         shuffle: bool = False,
#         logging: bool = False):
#         if (mode not in ['online', 'offline']): return []
#         if (rprop and mode != 'offline'): return []

#         online = mode == 'online'
#         train_data_x_orig = train_df_x.values.tolist()
#         train_data_y_orig = train_df_y.values.tolist()
#         learning_rate = default_learning_rate

#         eta_0 = 0.1
#         eta_max = 50
#         eta_min = 0.1
#         eta_p = 1.1
#         eta_n = 0.5

#         Err_hist = []
#         delta_w_prev = []
#         grad_prev = []
#         eta_prev = []

#         # iterate over epochs
#         for epoch in range(1, epochs + 1):
#             Err_e = 0.0
#             train_data_x = train_data_x_orig.copy()
#             grad = []
#             eta = []
#             w = []
            
#             # shuffle training set
#             if epoch > 1 and shuffle: rnd.shuffle(train_data_x)

#             # iterate of all training sets
#             for p_index, p in enumerate(train_data_x):
#                 y = self.predict(p) # output vector
#                 t = train_data_y_orig[train_data_x_orig.index(p)] # training input

#                 # error vector
#                 E_p = []               
#                 for y_index, y_j in enumerate(y):
#                     t_j = t[y_index] if type(t) == list else t
#                     E_p_y = t_j - y_j
#                     E_p.append(E_p_y)

#                 # specific error
#                 Err = 0.5 * sum([e * e for e in E_p])
#                 Err_e += (Err / len(train_data_x))

#                 # compute weight changes
#                 delta_w = []
                
#                 for layer_index in range(len(self.layers) - 1, 0, -1):
#                     is_output_layer = layer_index == len(self.layers) - 1

#                     neurons_h = self.layers[layer_index].neurons # current layer neurons
#                     weights_h = self.layers[layer_index].weights # current layer weights

#                     neurons_k = self.layers[layer_index - 1].neurons.copy() # previous layer neurons
#                     neurons_k.insert(0, self.layers[layer_index].bias_neuron) # add bias neuron of current layer to previous layer

#                     neurons_l = None if is_output_layer else self.layers[layer_index + 1].neurons # following layer neurons
#                     weights_l = None if is_output_layer else self.layers[layer_index + 1].weights # following layer weights

#                     act_func = self.layers[layer_index].activation_function
#                     delta_w.insert(0, [])
#                     if p_index == 0: grad.insert(0, [])

#                     for h_index, h in enumerate(neurons_h):
#                         act_der = h.do_activate_der(act_func) + flatspot_elim_value

#                         # compute delta_h
#                         if is_output_layer: delta_h = act_der * E_p[h_index]
#                         else:
#                             delta_sum = 0.0
#                             for l_index, l in enumerate(neurons_l):
#                                 w_h_l = weights_l[h_index + 1][l_index]
#                                 delta_sum += (l.delta * w_h_l)
#                             delta_h = act_der * delta_sum
#                         h.delta = delta_h

#                         delta_w[0].append([])
#                         if p_index == 0: grad[0].append([])
                        
#                         if not rprop:
#                             # compute delta w
#                             for k_index, k in enumerate(neurons_k):
#                                 if delta_w_prev: delta_w_k_h = learning_rate * k.output * delta_h + momentum_factor * delta_w_prev[layer_index - 1][h_index][k_index]
#                                 else: delta_w_k_h = learning_rate * k.output * delta_h
#                                 delta_w_k_h_prime = delta_w_k_h - learning_rate * weight_decay_factor * weights_h[k_index][h_index] # weight decay
#                                 delta_w[0][-1].append(delta_w_k_h_prime)
#                         else:
#                             # compute g
#                             for k_index, k in enumerate(neurons_k):
#                                 g_k_h = k.output * delta_h / len(train_data_x)
#                                 if p_index == 0: grad[0][-1].append(g_k_h)
#                                 else: grad[layer_index - 1][h_index][k_index] += g_k_h

#                 # process weight changes
#                 if not rprop:
#                     if online: # if training online, update weights
#                         for layer_index, l in enumerate(self.layers):
#                             if layer_index > 0:
#                                 delta_w_t = list(map(list, zip(*delta_w[layer_index - 1]))) # transpose weight matrix
#                                 for weight_r in range(len(l.weights)):
#                                     for weight_c in range(len(l.weights[0])):
#                                         l.weights[weight_r][weight_c] += delta_w_t[weight_r][weight_c]
#                         delta_w_prev = delta_w.copy()
#                     else: # if training offline, add weight change
#                         if p_index == 0: delta_w_o = delta_w.copy()
#                         else:
#                             for layer_index, l in enumerate(delta_w_o):
#                                 for n_index, n in enumerate(l):
#                                     for w_index in range(len(n)):
#                                         delta_w_o[layer_index][n_index][w_index] += delta_w[layer_index][n_index][w_index] / len(train_data_x)

#             # for offline learning, update weights after each epoch
#             if not online and not rprop:
#                 for layer_index, l in enumerate(self.layers):
#                     if layer_index > 0:
#                         delta_w_t = list(map(list, zip(*delta_w_o[layer_index - 1]))) # transpose weight matrix
#                         for weight_r in range(len(l.weights)):
#                             for weight_c in range(len(l.weights[0])):
#                                 l.weights[weight_r][weight_c] += delta_w_t[weight_r][weight_c] - learning_rate * weight_decay_factor * l.weights[weight_r][weight_c] # weight decay
#                 delta_w_prev = delta_w_o.copy()
#             elif rprop:
#                 for layer_index, l in enumerate(self.layers):
#                     if layer_index > 0:
#                         grad_t = list(map(list, zip(*grad[layer_index - 1]))) # transpose gradient matrix
#                         eta.append([])
#                         w.append([])

#                         if epoch > 1: grad_prev_t = list(map(list, zip(*grad_prev[layer_index - 1]))) # transpose previous gradient matrix

#                         for weight_r in range(len(l.weights)):
#                             eta[layer_index - 1].append([])
#                             w[layer_index - 1].append([])
#                             for weight_c in range(len(l.weights[0])):
#                                 # compute eta_k_h
#                                 if epoch == 1:
#                                     eta_k_h = eta_0
#                                 else:
#                                     if grad_t[weight_r][weight_c] * grad_prev_t[weight_r][weight_c] < 0.0: eta_k_h = eta_n * eta_prev[layer_index - 1][weight_r][weight_c]
#                                     elif grad_t[weight_r][weight_c] * grad_prev_t[weight_r][weight_c] > 0.0: eta_k_h = eta_p * eta_prev[layer_index - 1][weight_r][weight_c]
#                                     else: eta_k_h = eta_prev[layer_index - 1][weight_r][weight_c]
                                
#                                 if eta_k_h < eta_min: eta_k_h = eta_min
#                                 if eta_k_h > eta_max: eta_k_h = eta_max

#                                 # compute delta_w_k_h
#                                 if grad_t[weight_r][weight_c] > 0.0: delta_w_k_h = -eta_k_h
#                                 elif grad_t[weight_r][weight_c] < 0.0: delta_w_k_h = eta_k_h
#                                 else: delta_w_k_h = 0.0

#                                 # Save eta to list
#                                 eta[layer_index - 1][weight_r].append(eta_k_h)
#                                 w[layer_index - 1][weight_r].append(delta_w_k_h)

#                                 # update weight
#                                 l.weights[weight_r][weight_c] += delta_w_k_h
                                
#             eta_prev = eta.copy()
#             grad_prev = grad.copy()

#             # add error of epoch to list
#             Err_hist.append(Err_e)

#             # adapt learning rate based on error
#             error_dynamic = 0.0
#             if adaptive_learning_rate:
#                 error_dynamic = hlp.get_error_dynamic(Err_hist, 10)
#                 if  error_dynamic > 0.0: learning_rate /= 2.0

#             if logging:
#                 print(f'epoch = {epoch} | epoch error = {Err_e} | error dynamic = {error_dynamic} | learning rate = {learning_rate}')

#             if Err_hist[-1] < max_error:
#                 print(f'Training finished. Max error rate of < {max_error} reached on epoch {epoch}.')
#                 break

#             if learning_rate < min_learning_rate:
#                 print(f'Training aborted. Learning rate reached < {min_learning_rate} on epoch {epoch}.')
#                 break
        
#             if epoch == epochs:
#                 print(f'Training finished. Number of epochs reached.')

#         return Err_hist