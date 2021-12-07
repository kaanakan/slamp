import torch.nn as nn
import torch
from torch.autograd import Variable


class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.

        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))


class conv_lstm(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, n_layers, batch_size, image_size, device = torch.device('cpu')):
        super(conv_lstm, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.batch_size = batch_size
        self.image_size = image_size
        self.device = device
        self.embed = nn.Conv2d(input_size, hidden_size, 3, 1, 1)
        self.lstm = nn.ModuleList([ConvLSTMCell(hidden_size, hidden_size, (3, 3), True) for _ in range(self.n_layers)])
        self.output = nn.Sequential(
                nn.Conv2d(hidden_size, output_size, 3, 1, 1),
                nn.Tanh())
        self.hidden = self.init_hidden()

    def init_hidden(self, device=torch.device('cpu')):
        hidden = []
        for i in range(self.n_layers):
            hidden.append(self.lstm[i].init_hidden(self.batch_size, self.image_size))
        return hidden

    def forward(self, inp):
        embedded = self.embed(inp)
        h_in = embedded
        for i in range(self.n_layers):
            self.hidden[i] = self.lstm[i](h_in, self.hidden[i])
            h_in = self.hidden[i][0]
        return self.output(h_in)


class gaussian_conv_lstm(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, n_layers, batch_size, image_size, device=torch.device('cpu')):
        super(gaussian_conv_lstm, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.batch_size = batch_size
        self.image_size = image_size
        self.device = device
        self.embed = nn.Conv2d(input_size, hidden_size, 3, 1, 1)
        self.lstm = nn.ModuleList([ConvLSTMCell(input_size, hidden_size, (3, 3), True) for i in range(self.n_layers)])
        self.mu_net = nn.Conv2d(hidden_size, output_size, 3, 1, 1)
        self.logvar_net = nn.Conv2d(hidden_size, output_size, 3, 1, 1)
        self.hidden = self.init_hidden()

    def init_hidden(self, device=torch.device('cpu')):
        return [self.lstm[i].init_hidden(self.batch_size, self.image_size) for i in range(self.n_layers)]

    def reparameterize(self, mu, logvar):
        logvar = logvar.mul(0.5).exp_()
        eps = Variable(logvar.data.new(logvar.size()).normal_())
        return eps.mul(logvar).add_(mu)

    def forward(self, input):
        h_in = input
        for i in range(self.n_layers):
            self.hidden[i] = self.lstm[i](h_in, self.hidden[i])
            h_in = self.hidden[i][0]
        mu = self.mu_net(h_in)
        logvar = self.logvar_net(h_in)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar
