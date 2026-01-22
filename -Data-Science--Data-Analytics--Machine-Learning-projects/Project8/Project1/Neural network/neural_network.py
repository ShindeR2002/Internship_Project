import numpy as np

# --- 1. CORE ENGINE CLASSES ---
class Layer:
    def __init__(self):
        self.input_ = None 
        self.output_ = None 
    def forward_prop(self, input_): raise NotImplementedError
    def backward_prop(self, error, learning_rate, optimizer, t, regu): raise NotImplementedError

class Optimizer:
    def ADAM(self, dw, m, v, beta1, beta2):
        m = beta1 * m + (1 - beta1) * dw
        v = beta2 * v + (1 - beta2) * dw**2
        return m, v

class FC_layer(Layer):
    def __init__(self, input_size, output_size):
        np.random.seed(101)
        # He Initialization: Critical for 97%+ accuracy
        self.weights_ = np.random.randn(output_size, input_size) * np.sqrt(2 / input_size)
        self.bias_ = np.zeros((1, output_size))
        self.m_w, self.v_w = np.zeros_like(self.weights_), np.zeros_like(self.weights_)
        self.m_b, self.v_b = np.zeros_like(self.bias_), np.zeros_like(self.bias_)
        
    def forward_prop(self, input_):
        self.input_ = input_
        self.output_ = self.input_ @ self.weights_.T + self.bias_
        return self.output_
    
    def backward_prop(self, error, learning_rate, optimizer, t, regu): 
        lamda, _ = regu
        dw = error.T @ self.input_ + 2 * lamda * self.weights_
        db = np.sum(error, axis=0, keepdims=True)
        input_error = error @ self.weights_

        if optimizer == "ADAM":
            beta1, beta2, tol = 0.9, 0.999, 1e-08
            self.m_w, self.v_w = Optimizer().ADAM(dw, self.m_w, self.v_w, beta1, beta2)
            self.m_b, self.v_b = Optimizer().ADAM(db, self.m_b, self.v_b, beta1, beta2)
            m_hat_w, v_hat_w = self.m_w / (1 - beta1**t), self.v_w / (1 - beta2**t)
            m_hat_b, v_hat_b = self.m_b / (1 - beta1**t), self.v_b / (1 - beta2**t)
            self.weights_ -= (learning_rate * m_hat_w) / (np.sqrt(v_hat_w) + tol)
            self.bias_ -= (learning_rate * m_hat_b) / (np.sqrt(v_hat_b) + tol)
        return input_error

class AC_layer(Layer):
    def __init__(self, activation, activation_dr):
        self.act, self.act_dr = activation, activation_dr
    def forward_prop(self, input_):
        self.input_ = input_
        return self.act(self.input_)
    def backward_prop(self, error, learning_rate, optimizer, t, regu):
        return error * self.act_dr(self.input_)


class Dropout_layer(Layer):
    def __init__(self, probability):
        self.prob = probability
        self.mask = None

    def forward_prop(self, input_):
        # Create a mask of 1s and 0s based on probability
        self.mask = (np.random.rand(*input_.shape) > self.prob) / (1.0 - self.prob)
        return input_ * self.mask

    def backward_prop(self, error, learning_rate, optimizer, t, regu):
        # Only propagate error through the neurons that were 'on'
        return error * self.mask


class Activation:
    def relu(self, x): return np.maximum(0, x)
    def relu_prime(self, x): return (x > 0).astype(float)
    def softmax(self, z):
        shift_z = z - np.max(z, axis=1, keepdims=True)
        exps = np.exp(shift_z)
        return exps / np.sum(exps, axis=1, keepdims=True)
    def softmax_dr(self, z):
        s = self.softmax(z)
        return s * (1 - s)

class Network:
    def __init__(self):
        self.layers, self.error_list = [], []
    def add(self, layer): self.layers.append(layer)
    def path(self, neuron_info, dropout_rate=0.2):
        for i in range(1, len(neuron_info)):
            self.add(FC_layer(neuron_info[i-1], neuron_info[i]))
            if i == len(neuron_info) - 1:
                self.add(AC_layer(Activation().softmax, Activation().softmax_dr))
            else:
                self.add(AC_layer(Activation().relu, Activation().relu_prime))
                # Add dropout after each hidden activation
                self.add(Dropout_layer(dropout_rate))
    def predict(self, X):
        output = X
        for layer in self.layers: output = layer.forward_prop(output)
        return output
    def fit(self, X, y, epoch, learning_rate, batch_size, optimizer, loss_type, regularization):
        l2_lamda, _ = regularization
        t, m = 0, X.shape[0]
        # Clear any existing errors before a new training run
        self.error_list = [] 
        
        for i in range(epoch):
            for batch in range(m // batch_size):
                t += 1
                idx = slice(batch * batch_size, (batch + 1) * batch_size)
                output = self.predict(X[idx])
                err = (output - y[idx]) / batch_size
                for layer in reversed(self.layers):
                    err = layer.backward_prop(err, learning_rate, optimizer, t, (l2_lamda, 0))
            
            # This line populates the data for your Convergence Plot
            current_err = np.linalg.norm(self.predict(X[:1000]) - y[:1000])
            self.error_list.append(current_err)
            
            if i % 5 == 0 and i > 0: learning_rate *= 0.8

# --- 2. THE PROTECTIVE BLOCK ---
# This ensures code inside only runs when you execute this file directly.
# It prevents the NameError when you import into your notebook.
if __name__ == "__main__":
    print("Neural Network Engine loaded successfully in standalone mode.")