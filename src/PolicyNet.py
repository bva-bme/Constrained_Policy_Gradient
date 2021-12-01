import torch


class NeuralNet(torch.nn.Module):

    def __init__(self, input_size, output_size, hidden_size, lr=1e-4):
        super(NeuralNet, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.HiddenLayer = torch.nn.Linear(self.input_size, self.hidden_size, bias=True).double()
        self.DropoutLayer = torch.nn.Dropout(p=0.0)
        self.OutputLayer = torch.nn.Linear(self.hidden_size, self.output_size, bias=True).double()

        self.neural_tangent_kernel = None

        # For training
        self.learning_rate = lr
        self.optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate)
        self.loss_arr = []
        self.__policy_loss = []

    def forward(self, x):
        x = x.double()
        x = self.HiddenLayer(x)  # b*x
        x = self.DropoutLayer(x)
        x = torch.relu(x)  # sigma(b*x)
        x = self.OutputLayer(x)   # y = a*sigma(b*x) + c
        x = torch.softmax(x, dim=1)
        return x  # probabilities

    def __partial_derivatives(self, x):
        self.zero_grad()

        w1 = torch.empty(self.output_size, self.hidden_size*self.input_size + self.hidden_size, dtype=torch.float64)
        w2 = torch.empty(self.output_size, self.hidden_size*self.output_size + self.output_size, dtype=torch.float64)

        for i in range(self.output_size):
            y = self.forward(x)
            y = y[0][i]
            y.backward()  # handle multiple outputs

            wi1 = self.HiddenLayer.weight.grad  # nabla_theta f(x) (Jacobian)
            wi1 = torch.reshape(wi1, [wi1.shape[0] * wi1.shape[1], 1])
            wi1 = torch.cat([wi1, self.HiddenLayer.bias.grad.unsqueeze(1)])

            wi2 = self.OutputLayer.weight.grad
            wi2 = torch.reshape(wi2, [wi2.shape[0] * wi2.shape[1], 1])
            wi2 = torch.cat([wi2, self.OutputLayer.bias.grad.unsqueeze(1)])

            wi1g = wi1.clone().detach()
            wi2g = wi2.clone().detach()  # create deep copy, otherwise gradients keep rolling

            w1[i] = wi1g.squeeze()
            w2[i] = wi2g.squeeze()

            self.zero_grad()

        return w1, w2

    def compute_neural_tangent_kernel(self, x):

        kernel = torch.zeros([x.shape[0] * self.output_size, x.shape[0] * self.output_size], dtype=torch.float64,
                             requires_grad=False)

        i = 0
        for x1 in x.data:
            w1x1, w2x1 = self.__partial_derivatives(x1.unsqueeze(dim=0))
            j = 0
            for x2 in x.data:
                # sum_{i=1}^m  (df(x1)/dw1)^T*(df(x2)/dw1) + ...
                w1x2, w2x2 = self.__partial_derivatives(x2.unsqueeze(dim=0))
                kernel[self.output_size * i:self.output_size * i + self.output_size,
                       self.output_size * j:self.output_size * j + self.output_size] = \
                    torch.matmul(w1x1, w1x2.transpose(0, 1)) + torch.matmul(w2x1, w2x2.transpose(0, 1))
                j += 1
            i += 1

        self.neural_tangent_kernel = kernel

        return kernel

    def train_network(self, log_probs, gains, gains_normed=0):

        self.__policy_loss = []
        eps = 1E-8
        batch_len = len(gains)
        gains.clone().detach().requires_grad_(True)

        if gains_normed:
            gains_norm = gains
        else:
            gains_norm = (gains - gains.mean()) / (gains.std() + eps)  # normalize
        for k in range(batch_len):
            self.__policy_loss.append(-log_probs[k] * gains_norm[k])  # norm

        self.__policy_loss = torch.cat(self.__policy_loss).sum()
        self.__policy_loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.loss_arr.append(self.__policy_loss.item())
