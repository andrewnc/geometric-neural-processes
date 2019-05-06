import numpy as np
import matplotlib.pyplot as plt
import collections
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import time
from tqdm import tqdm
import utils

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

# The (A)NP takes as input a `NPRegressionDescription` namedtuple with fields:
#   `query`: a tuple containing ((context_x, context_y), target_x)
#   `target_y`: a tensor containing the ground truth for the targets to be
#     predicted
#   `num_total_points`: A vector containing a scalar that describes the total
#     number of datapoints used (context + target)
#   `num_context_points`: A vector containing a scalar that describes the number
#     of datapoints used as context
# The GPCurvesReader returns the newly sampled data in this format at each
# iteration

NPRegressionDescription = collections.namedtuple(
    "NPRegressionDescription",
    ("query", "target_y", "num_total_points", "num_context_points"))

# TODO: Check the tiling operation, since I have to write it myself.
def tile(x, multiples):
    """
    PyTorch implementation of tf.tile(). See https://stackoverflow.com/a/52259068.
    Constructs a tensor by tiling a given tensor.

    This operation creates a new tensor by replicating input multiples times.
    The output tensor's i'th dimension has input.dims(i) * multiples[i] elements, and the values of input are
    replicated multiples[i] times along the 'i'th dimension.
    For example, tiling [a b c d] by [2] produces [a b c d a b c d].
    """
    return x.repeat(multiples)


class GPCurvesReader(object):
    """Generates curves using a Gaussian Process (GP).

    Supports vector inputs (x) and vector outputs (y). Kernel is
    mean-squared exponential, using the x-value l2 coordinate distance scaled by
    some factor chosen randomly in a range. Outputs are independent gaussian
    processes.
    """

    def __init__(self,
               batch_size,
               max_num_context,
               x_size=1,
               y_size=1,
               l1_scale=0.6,
               sigma_scale=1.0,
               random_kernel_parameters=True,
               testing=False):
        """Creates a regression dataset of functions sampled from a GP.

        Args:
          batch_size: An integer.
          max_num_context: The max number of observations in the context.
          x_size: Integer >= 1 for length of "x values" vector.
          y_size: Integer >= 1 for length of "y values" vector.
          l1_scale: Float; typical scale for kernel distance function.
          sigma_scale: Float; typical scale for variance.
          random_kernel_parameters: If `True`, the kernel parameters (l1 and sigma)
              will be sampled uniformly within [0.1, l1_scale] and [0.1, sigma_scale].
          testing: Boolean that indicates whether we are testing. If so there are
              more targets for visualization.
        """
        self._batch_size = batch_size
        self._max_num_context = max_num_context
        self._x_size = x_size
        self._y_size = y_size
        self._l1_scale = l1_scale
        self._sigma_scale = sigma_scale
        self._random_kernel_parameters = random_kernel_parameters
        self._testing = testing

    def _gaussian_kernel(self, xdata, l1, sigma_f, sigma_noise=2e-2):
        """Applies the Gaussian kernel to generate curve data.

        Args:
          xdata: Tensor of shape [B, num_total_points, x_size] with
              the values of the x-axis data.
          l1: Tensor of shape [B, y_size, x_size], the scale
              parameter of the Gaussian kernel.
          sigma_f: Tensor of shape [B, y_size], the magnitude
              of the std.
          sigma_noise: Float, std of the noise that we add for stability.

        Returns:
          The kernel, a float tensor of shape
          [B, y_size, num_total_points, num_total_points].
        """
        num_total_points = xdata.shape[1]

        # Expand and take the difference
        xdata1 = xdata.unsqueeze(1) # [B, 1, num_total_points, x_size]
        xdata2 = xdata.unsqueeze(2) # [B, num_total_points, 1, x_size]
        diff = xdata1 - xdata2  # [B, num_total_points, num_total_points, x_size]

        norm = (diff[:, None, :, :, :] / l1[:, :, None, None, :]) ** 2
        norm = norm.sum(dim=-1) # [B, data_size, num_total_points, num_total_points]

        kernel = (sigma_f ** 2)[:, :, None, None] * torch.exp(-0.5 * norm)
        # Add some noise to the diagonal to make the cholesky work.
        kernel += (sigma_noise ** 2) * torch.eye(num_total_points)

        return kernel

    def generate_curves(self):
        """Builds the op delivering the data.

        Generated functions are `float32` with x values between -2 and 2.

        Returns:
          A `CNPRegressionDescription` namedtuple.
        """
        num_context = torch.empty(1).uniform_(3, self._max_num_context).int().item()

        # If we are testing we want to have more targets and have them evenly
        # distributed in order to plot the function.
        if self._testing:
            num_target = 400
            num_total_points = num_target
            x_values = tile(
            torch.arange(-2, 2, 0.01).unsqueeze(0),
            (self._batch_size, 1)
            )
            x_values = x_values.unsqueeze(-1)
        # During training the number of target points and their x-positions are
        # selected at random
        else:
            num_target = torch.randint(0, self._max_num_context - num_context, size=())
            num_total_points = num_context + num_target
            x_values = torch.FloatTensor(self._batch_size, num_total_points, self._x_size).uniform_(-2, 2)

        # Set kernel parameters
        # Either choose a set of random parameters for the mini-batch
        if self._random_kernel_parameters:
            l1 = torch.FloatTensor(self._batch_size, self._y_size, self._x_size).uniform_(0.1, self._sigma_scale)
            sigma_f = torch.FloatTensor(self._batch_size, self._y_size).uniform_(0.1, self._sigma_scale)
        # Or use the same fixed parameters for all mini-batches
        else:
            l1 = torch.ones(self._batch_size, self._y_size, self._x_size) * self._l1_scale
            sigma_f = torch.ones(self._batch_size, self._y_size) * self._sigma_scale

        # Pass the x_values through the Gaussian kernel
        kernel = self._gaussian_kernel(x_values, l1, sigma_f) # [batch_size, y_size, num_total_points, num_total_points] afterwards

        # Calculate Cholesky, using double precision for better stability:
        cholesky = torch.cholesky(kernel.double()).float()

        # Sample a curve
        # [batch_size, y_size, num_total_points, 1]
        y_values = cholesky @ torch.randn(self._batch_size, self._y_size, num_total_points, 1)

        # [batch_size, num_total_points, y_size]
        y_values = y_values.squeeze(3).permute(0, 2, 1)

        if self._testing:
            # Select the targets
            target_x = x_values
            target_y = y_values

            # Select the observations
            idx = torch.randperm(num_target)
            idx_context = idx[:num_context].unsqueeze(0).unsqueeze(2) # torch.gather differs from tf.gather
            context_x = torch.gather(input=x_values, dim=1, index=idx_context)
            context_y = torch.gather(input=y_values, dim=1, index=idx_context)

        else:
            # Select the targets which will consist of the context points as well as
            # some new target points
            target_x = x_values[:, :num_target + num_context, :]
            target_y = y_values[:, :num_target + num_context, :]

            # Select the observations
            context_x = x_values[:, :num_context, :]
            context_y = y_values[:, :num_context, :]

        query = ((context_x.to(device), context_y.to(device)), target_x.to(device))

        return NPRegressionDescription(
        query=query,
        target_y=target_y.to(device),
        num_total_points=target_x.shape[1],
        num_context_points=num_context)

# Pytorch doesn't recognize a list of nn.Modules, so we have to use the nn.ModuleList construct
class BatchMLP(nn.Module):
    def __init__(self, xy_size, output_sizes):
        super().__init__()
        self.output_sizes = output_sizes

        self.layers = nn.ModuleList([nn.Linear(xy_size, output_sizes[0])] + [
            nn.Linear(output_sizes[i-1], output_sizes[i])
            for i in range(1, len(output_sizes))
        ])


    def forward(self, x):
        # Get the shapes of the input and reshape to parallelise across observations
        batch_size, _, xy_size = x.shape
        x = x.contiguous().view(-1, xy_size)

        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        x = self.layers[-1](x)

        # Bring back into original shape
        x = x.view(batch_size, -1, self.output_sizes[-1])
        return x

class DeterministicEncoder(nn.Module):
    """The Deterministic Encoder."""

    def __init__(self, xy_size, output_sizes, attention):
        """(A)NP deterministic encoder.

        Args:
          output_sizes: An iterable containing the output sizes of the encoding MLP.
          attention: The attention module.
        """
        super().__init__()
        self._output_sizes = output_sizes
        self._attention = attention

        self._mlp = BatchMLP(xy_size, output_sizes)

    def forward(self, context_x, context_y, target_x):
        """Encodes the inputs into one representation.

        Args:
          context_x: Tensor of shape [B,observations,d_x]. For this 1D regression
              task this corresponds to the x-values.
          context_y: Tensor of shape [B,observations,d_y]. For this 1D regression
              task this corresponds to the y-values.
          target_x: Tensor of shape [B,target_observations,d_x].
              For this 1D regression task this corresponds to the x-values.

        Returns:
          The encoded representation. Tensor of shape [B,target_observations,d]
        """

        # Concatenate x and y along the filter axes
        encoder_input = torch.cat([context_x, context_y], dim=-1)

        # Pass final axis through MLP
        hidden = self._mlp(encoder_input)

        # Apply attention
        hidden = self._attention(context_x, target_x, hidden)

        return hidden

# We have varying input dimensions - [B, observations, d_xandy]
class LatentEncoder(nn.Module):
    """The Latent Encoder."""

    def __init__(self, xy_size, output_sizes, num_latents):
        """(A)NP latent encoder.

        Args:
          output_sizes: An iterable containing the output sizes of the encoding MLP.
          num_latents: The latent dimensionality.
        """
        super().__init__()
        self._output_sizes = output_sizes
        self._num_latents = num_latents
        self._xy_size = xy_size

        self._mlp = BatchMLP(xy_size=xy_size, output_sizes=output_sizes)

        n_hidden = (self._output_sizes[-1] + self._num_latents) // 2
        self.penultimate = nn.Linear(output_sizes[-1], n_hidden)
        self.mean = nn.Linear(n_hidden, self._num_latents)
        self.log_sigma = nn.Linear(n_hidden, self._num_latents)

    def forward(self, x, y):
        """Encodes the inputs into one representation.

        Args:
          x: Tensor of shape [B,observations,d_x]. For this 1D regression
              task this corresponds to the x-values.
          y: Tensor of shape [B,observations,d_y]. For this 1D regression
              task this corresponds to the y-values.

        Returns:
          A normal distribution over tensors of shape [B, num_latents]
        """

        # Concatenate x and y along the filter axes
        encoder_input = torch.cat([x, y], dim=-1)

        # Pass final axis through MLP
        hidden = self._mlp(encoder_input)

        # Aggregator: take the mean over all points
        hidden = torch.mean(hidden, dim=1)

        # Have further MLP layers that map to the parameters of the Gaussian latent
        hidden = F.relu(self.penultimate(hidden))
        mu = self.mean(hidden)
        log_sigma = self.log_sigma(hidden)

        # Compute sigma
        sigma = 0.1 + 0.9 * F.softplus(log_sigma) # CHANGE: Original had a sigmoid instead, but that led to vanishing gradients

        return torch.distributions.Normal(loc=mu, scale=sigma)

class Decoder(nn.Module):
    """The Decoder."""

    def __init__(self, input_size, output_sizes):
        """(A)NP decoder.

        Args:
          output_sizes: An iterable containing the output sizes of the decoder MLP
              as defined in `basic.Linear`.
        """
        super().__init__()
        self._output_sizes = output_sizes
        self._mlp = BatchMLP(xy_size=input_size, output_sizes=output_sizes)

    def forward(self, representation, target_x):
        """Decodes the individual targets.

        Args:
          representation: The representation of the context for target predictions.
              Tensor of shape [B,target_observations,?].
          target_x: The x locations for the target query.
              Tensor of shape [B,target_observations,d_x].

        Returns:
          dist: A multivariate Gaussian over the target points. A distribution over
              tensors of shape [B,target_observations,d_y].
          mu: The mean of the multivariate Gaussian.
              Tensor of shape [B,target_observations,d_x]. # TODO: Should this be d_y?
          sigma: The standard deviation of the multivariate Gaussian.
              Tensor of shape [B,target_observations,d_x]. # TODO: Should this be d_y?
        """
        # concatenate target_x and representation
        hidden = torch.cat([representation, target_x], dim=-1)

        # Pass final axis through MLP
        hidden = self._mlp(hidden)

        # Get the mean an the variance
        mu, log_sigma = torch.split(hidden, hidden.shape[-1] // 2, dim=-1) # Split the last dimension in two

        # Bound the variance
        sigma = 0.1 + 0.9 * F.softplus(log_sigma)
        # mu, sigma are each [batch, n_observations, 1] tensors

        # Get the distribution
        sigma_diag = torch.diag_embed(sigma.squeeze(-1)) # [batch, n_observations, n_observations]
        sigma_diag = torch.diag_embed(sigma) # [batch, n_observations, d_y, d_y]
        dist = torch.distributions.MultivariateNormal(
            loc=mu, scale_tril=sigma_diag)

        return dist, mu, sigma

class LatentModel(nn.Module):
    """The (A)NP model."""

    def __init__(self, x_size, y_size, latent_encoder_output_sizes, num_latents,
               decoder_output_sizes, use_deterministic_path=True,
               deterministic_encoder_output_sizes=None, attention=None):
        """Initialises the model.

        Args:
          latent_encoder_output_sizes: An iterable containing the sizes of hidden
              layers of the latent encoder.
          num_latents: The latent dimensionality.
          decoder_output_sizes: An iterable containing the sizes of hidden layers of
              the decoder. The last element should correspond to d_y * 2
              (it encodes both mean and variance concatenated)
          use_deterministic_path: a boolean that indicates whether the deterministic
              encoder is used or not.
          deterministic_encoder_output_sizes: An iterable containing the sizes of
              hidden layers of the deterministic encoder. The last one is the size
              of the deterministic representation r.
          attention: The attention module used in the deterministic encoder.
              Only relevant when use_deterministic_path=True.
        """
        super().__init__()
        self._xy_size = x_size + y_size

        self._latent_encoder = LatentEncoder(xy_size=self._xy_size, output_sizes=latent_encoder_output_sizes,
                                             num_latents=num_latents)
        self._use_deterministic_path = use_deterministic_path
        if use_deterministic_path:
            self._deterministic_encoder = DeterministicEncoder(
              self._xy_size, deterministic_encoder_output_sizes, attention)

        decoder_input_size = latent_encoder_output_sizes[-1] + x_size
        if use_deterministic_path:
            decoder_input_size += deterministic_encoder_output_sizes[-1]
        self._decoder = Decoder(decoder_input_size, decoder_output_sizes)



    def forward(self, query, num_targets, target_y=None):
        """Returns the predicted mean and variance at the target points.

        Args:
          query: Array containing ((context_x, context_y), target_x) where:
              context_x: Tensor of shape [B,num_contexts,d_x].
                  Contains the x values of the context points.
              context_y: Tensor of shape [B,num_contexts,d_y].
                  Contains the y values of the context points.
              target_x: Tensor of shape [B,num_targets,d_x].
                  Contains the x values of the target points.
          num_targets: Number of target points.
          target_y: The ground truth y values of the target y.
              Tensor of shape [B,num_targets,d_y].

        Returns:
          log_p: The log_probability of the target_y given the predicted
              distribution. Tensor of shape [B,num_targets].
          mu: The mean of the predicted distribution.
              Tensor of shape [B,num_targets,d_y].
          sigma: The variance of the predicted distribution.
              Tensor of shape [B,num_targets,d_y].
        """

        (context_x, context_y), target_x = query

        # Pass query through the encoder and the decoder
        prior = self._latent_encoder(context_x, context_y)

        # For training, when target_y is available, use targets for latent encoder.
        # Note that targets contain contexts by design.
        if target_y is None:
            latent_rep = prior.rsample()
        # For testing, when target_y unavailable, use contexts for latent encoder.
        else:
            posterior = self._latent_encoder(target_x, target_y)
            latent_rep = posterior.rsample() # CHANGE: I used rsample() instead of sample() for better gradient flow

        latent_rep = tile(latent_rep.unsqueeze(1), [1, num_targets, 1])
        if self._use_deterministic_path:
            deterministic_rep = self._deterministic_encoder(context_x, context_y, target_x)
            representation = torch.cat((deterministic_rep, latent_rep), dim=-1)
        else:
            representation = latent_rep

        dist, mu, sigma = self._decoder(representation, target_x)

        # If we want to calculate the log_prob for training we will make use of the
        # target_y. At test time the target_y is not available so we return None.
        if target_y is not None:
            log_p = dist.log_prob(target_y)
            posterior = self._latent_encoder(target_x, target_y)
            kl = torch.distributions.kl.kl_divergence(posterior, prior).sum(dim=-1, keepdim=True)
            kl = tile(kl, [1, num_targets])
            loss = - torch.mean(log_p - kl / num_targets)
        else:
            log_p = None
            kl = None
            loss = None

        return mu, sigma, log_p, kl, loss

def uniform_attention(q, v):
    """uniform attention. equivalent to np.

    args:
    q: queries. tensor of shape [b,m,d_k].
    v: values. tensor of shape [b,n,d_v].

    returns:
    tensor of shape [b,m,d_v].
    """
    total_points = q.shape[1]
    rep = v.mean(dim=1, keepdim=true)
    rep = tile(rep, [1, total_points, 1])
    return rep

def laplace_attention(q, k, v, scale, normalise):
    """computes laplace exponential attention.

    args:
    q: queries. tensor of shape [b,m,d_k].
    k: keys. tensor of shape [b,n,d_k].
    v: values. tensor of shape [b,n,d_v].
    scale: float that scales the l1 distance.
    normalise: boolean that determines whether weights sum to 1.

    returns:
    tensor of shape [b,m,d_v].
    """
    k = k.unsqueeze(1)
    q = q.unsqueeze(2)
    unnorm_weights = -torch.abs((k - q) / scale)
    unnorm_weights = unnorm_weights.sum(dim=-1) #[b,m,n]
    if normalise:
        weight_fn = lambda x: f.softmax(x, dim=-1)
    else:
        weight_fn = lambda x: 1 + f.tanh(x)
    weights = weight_fn(unnorm_weights)  # [b,m,n]
    rep = torch.einsum('bik,bkj->bij', weights, v)
    return rep

def dot_product_attention(q, k, v, normalise):
    """computes dot product attention.

    args:
    q: queries. tensor of  shape [b,m,d_k].
    k: keys. tensor of shape [b,n,d_k].
    v: values. tensor of shape [b,n,d_v].
    normalise: boolean that determines whether weights sum to 1.

    returns:
    tensor of shape [b,m,d_v].
    """
    d_k = q.shape[-1]
    scale = np.sqrt(d_k)
    unnorm_weights = torch.einsum('bjk,bik->bij', k, q) / scale # [b, m, n]
    if normalise:
        weight_fn = lambda x: f.softmax(x, dim=-1)
    else:
        weight_fn = f.sigmoid
    weights = weight_fn(unnorm_weights)  # [b,m,n]
    rep = torch.einsum('bik,bkj->bij', weights, v)  # [b,m,d_v]
    return rep


def multihead_attention(q, k, v, wqs, wks, wvs, wo, num_heads=8):
    """computes multi-head attention.

    args:
    q: queries. tensor of  shape [b,m,d_k].
    k: keys. tensor of shape [b,n,d_k].
    v: values. tensor of shape [b,n,d_v].
    wqs: list of linear query transformation. [linear(?, d_k)]
    wks: list of linear key transformations. [linear(?, d_k), ...]
    wvs: list of linear value transformations. [linear(?, d_v), ...]
    wo: linear transformation for output of dot-product attention
    num_heads: number of heads. should divide d_v.

    returns:
    tensor of shape [b,m,d_v].
    """

    d_k = q.shape[-1]
    d_v = v.shape[-1]
    head_size = d_v / num_heads
    rep = 0

    for h in range(num_heads):
        q_h = wqs[h](q)
        k_h = wks[h](k)
        v_h = wvs[h](v)
        o = dot_product_attention(q_h, k_h, v_h, normalise=true)
        rep += wo(o)

    return rep


class attention(nn.module):
    """the attention module."""

    def __init__(self, rep, x_size, r_size, output_sizes, att_type, scale=1., normalise=true,
               num_heads=8):
        """create attention module.

        takes in context inputs, target inputs and
        representations of each context input/output pair
        to output an aggregated representation of the context data.
        args:
          rep: transformation to apply to contexts before computing attention.
              one of: ['identity','mlp'].
          output_sizes: list of number of hidden units per layer of mlp.
              used only if rep == 'mlp'.
          att_type: type of attention. one of the following:
              ['uniform','laplace','dot_product','multihead']
          scale: scale of attention.
          normalise: boolean determining whether to:
              1. apply softmax to weights so that they sum to 1 across context pts or
              2. apply custom transformation to have weights in [0,1].
          num_heads: number of heads for multihead.
        """
        super().__init__()
        self._rep = rep
        self._output_sizes = output_sizes
        self._type = att_type
        self._scale = scale
        self._normalise = normalise

        d_v = r_size
        if self._rep =='mlp':
            self._mlp = batchmlp(xy_size=x_size, output_sizes=output_sizes)
            d_k = output_sizes[-1] # dimension of keys and queries
        else:
            d_k = x_size

        if self._type == 'multihead':
            head_size = d_v // num_heads
            self._num_heads = num_heads
            self._wqs = nn.modulelist([
              batchmlp(d_k, output_sizes=[head_size])
              for h in range(num_heads)
            ])
            self._wks = nn.modulelist([
              batchmlp(d_k, output_sizes=[head_size])
              for h in range(num_heads)
            ])
            self._wvs = nn.modulelist([
              batchmlp(d_v, output_sizes=[head_size])
              for h in range(num_heads)
            ])
            self._wo = batchmlp(head_size, [d_v])

    def forward(self, context_x, target_x, r):
        """apply attention to create aggregated representation of r.

        args:
          context_x: tensor of shape [b,n1,d_x] (keys)
          target_x: tensor of shape [b,n2,d_x] (queries)
          r: tensor of shape [b,n1,d] (values)

        returns:
          tensor of shape [b,n2,d]

        raises:
          nameerror: the argument for rep/type was invalid.
        """
        if self._rep == 'identity':
            target_x *= self._coef # this has grad
            k, q = context_x, target_x
        elif self._rep == 'mlp':
          # pass through mlp
            k = self._mlp(context_x)
            q = self._mlp(target_x)
        else:
            raise nameerror("'rep' not among ['identity','mlp']")

        if self._type == 'uniform':
            rep = uniform_attention(q, r)
        elif self._type == 'laplace':
            rep = laplace_attention(q, k, r, self._scale, self._normalise)
        elif self._type == 'dot_product':
            rep = dot_product_attention(q, k, r, self._normalise)
        elif self._type == 'multihead':
            rep = multihead_attention(q, k, r, self._wqs, self._wks, self._wvs, self._wo, self._num_heads)
        else:
            raise nameerror(("'att_type' not among ['uniform','laplace','dot_product'"
                           ",'multihead']"))

        return rep

def plot_functions(target_x, target_y, context_x, context_y, pred_y, std):
    """plots the predicted mean and variance and the context points.

    args:
        target_x: an array of shape [b,num_targets,1] that contains the
            x values of the target points.
        target_y: an array of shape [b,num_targets,1] that contains the
            y values of the target points.
        context_x: an array of shape [b,num_contexts,1] that contains
            the x values of the context points.
        context_y: an array of shape [b,num_contexts,1] that contains
            the y values of the context points.
        pred_y: an array of shape [b,num_targets,1] that contains the
            predicted means of the y values at the target points in target_x.
        std: an array of shape [b,num_targets,1] that contains the
            predicted std dev of the y values at the target points in target_x.
    """
    target_x, target_y = target_x.cpu().detach().numpy(), target_y.cpu().detach().numpy()
    context_x, context_y = context_x.cpu().detach().numpy(), context_y.cpu().detach().numpy()
    pred_y, std = pred_y.cpu().detach().numpy(), std.cpu().detach().numpy()

    # plot everything
    plt.plot(target_x[0], pred_y[0], 'b', linewidth=2)
    plt.plot(target_x[0], target_y[0], 'k:', linewidth=2)
    plt.plot(context_x[0], context_y[0], 'ko', markersize=10)
    plt.fill_between(
      target_x[0, :, 0],
      pred_y[0, :, 0] - std[0, :, 0],
      pred_y[0, :, 0] + std[0, :, 0],
      alpha=0.2,
      facecolor='#65c9f7',
      interpolate=true)

    # make the plot pretty
    plt.yticks([-2, 0, 2], fontsize=16)
    plt.xticks([-2, 0, 2], fontsize=16)
    plt.ylim([-2, 2])
    plt.grid(false)
    ax = plt.gca()
    plt.show()

def plot_quartet(ground_truth_image, sparse_data, mu, sigma, epoch, i, m, n):
    fig, ax = plt.subplots(ncols=2, nrows=2)
    ax[0][1].imshow(ground_truth_image.reshape(m,n).cpu())
    ax[0][0].imshow(sparse_data.reshape(m,n).cpu())
    ax[1][0].imshow(mu.detach().reshape(m,n).cpu())
    ax[1][1].imshow(sigma.detach().reshape(m,n).cpu())
    plt.savefig("attention{}res{}.png".format(epoch, i))
    plt.close()

def plot_grad_flow(named_parameters):
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            if p.grad is none:
#                 print("{} has no grad".format(n))
                continue
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
#             print("{}: {:.3e}".format(n, ave_grads[-1]))
    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.xlabel("layers")
    plt.ylabel("average gradient")
    plt.title("gradient flow")
    plt.grid(true)
    plt.show()

def split_target_context(ground_truth_image, image_frame, batch_size, X_SIZE, Y_SIZE, min_context_points, max_context_points):
    """[batch, c, m, n]"""
    sparse_data = utils.batch_context(ground_truth_image, context_points=np.random.randint(min_context_points, max_context_points)).float().to(device)

    context_x, context_y = utils.batch_features(sparse_data)
    context_x, context_y = context_x.to(device), context_y.to(device)

    target_x = image_frame
    target_x = target_x.view(batch_size,-1,X_SIZE)
    target_y = ground_truth_image
    target_y = target_y.view(batch_size,-1,Y_SIZE)

    target_x, target_y = target_x.to(device), target_y.to(device)
    return sparse_data, context_x, context_y, target_x, target_y



def train_regression():
    TRAINING_ITERATIONS = 100000 #@param {type:"number"}
    MAX_CONTEXT_POINTS = 50 #@param {type:"number"}
    PLOT_AFTER = 100 #10000 #@param {type:"number"}
    HIDDEN_SIZE = 128 #@param {type:"number"}
    MODEL_TYPE = 'ANP' #@param ['NP','ANP']
    ATTENTION_TYPE = 'multihead' #@param ['uniform','laplace','dot_product','multihead']
    random_kernel_parameters = True #@param {type:"boolean"}
    X_SIZE = 2
    Y_SIZE = 1
    batch_size = 1
    test_batch_size = 1


    # Train dataset
    # dataset_train = GPCurvesReader(
    #     batch_size=16, x_size=X_SIZE, y_size=Y_SIZE, max_num_context=MAX_CONTEXT_POINTS, random_kernel_parameters=random_kernel_parameters)
    # data_train = dataset_train.generate_curves()

    # # Test dataset
    # dataset_test = GPCurvesReader(
    #     batch_size=1, x_size=X_SIZE, y_size=Y_SIZE, max_num_context=MAX_CONTEXT_POINTS, testing=True, random_kernel_parameters=random_kernel_parameters)
    # data_test = dataset_test.generate_curves()

    kwargs = {'num_workers': 1, 'pin_memory': False} if True else {}
    # kwargs = {}

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                    ])),
        batch_size=batch_size, shuffle=True, **kwargs)


    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                        transforms.ToTensor(),
                    ])),
        batch_size=test_batch_size, shuffle=True, **kwargs)


    # Sizes of the layers of the MLPs for the encoders and decoder
    # The final output layer of the decoder outputs two values, one for the mean and
    # one for the variance of the prediction at the target location
    latent_encoder_output_sizes = [HIDDEN_SIZE]*4
    num_latents = HIDDEN_SIZE
    deterministic_encoder_output_sizes= [HIDDEN_SIZE]*4
    decoder_output_sizes = [HIDDEN_SIZE]*2 + [2]
    use_deterministic_path = True
    xy_size = X_SIZE + Y_SIZE
    m, n = 28, 28
    num_pixels = m*n

    min_context_points = num_pixels * 0.15 # always have at least 15% of all pixels
    max_context_points = num_pixels * 0.95 # always have at most 95% of all pixels


    # ANP with multihead attention
    if MODEL_TYPE == 'ANP':
        attention = Attention(rep='mlp', x_size=X_SIZE, r_size=deterministic_encoder_output_sizes[-1], output_sizes=[HIDDEN_SIZE]*2,
                            att_type=ATTENTION_TYPE).to(device) # CHANGE: rep was originally 'mlp'
    # NP - equivalent to uniform attention
    elif MODEL_TYPE == 'NP':
        attention = Attention(rep='identity', x_size=None, output_sizes=None, att_type='uniform').to(device)
    else:
        raise NameError("MODEL_TYPE not among ['ANP,'NP']")

    # Define the model
    print("num_latents: {}, latent_encoder_output_sizes: {}, deterministic_encoder_output_sizes: {}, decoder_output_sizes: {}".format(
        num_latents, latent_encoder_output_sizes, deterministic_encoder_output_sizes, decoder_output_sizes))
    decoder_input_size = 2 * HIDDEN_SIZE + X_SIZE
    model = LatentModel(X_SIZE, Y_SIZE, latent_encoder_output_sizes, num_latents,
                        decoder_output_sizes, use_deterministic_path,
                        deterministic_encoder_output_sizes, attention)

    # Define the loss
    model = model.to(device)

    image_frame = torch.tensor([[i, j] for i in range(0,m) for j in range(0,n)]).repeat(batch_size,Y_SIZE).float().to(device)
    test_image_frame = torch.tensor([[i, j] for i in range(0,m) for j in range(0,n)]).repeat(test_batch_size,Y_SIZE).float().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    for epoch in range(10):
        progress = tqdm(enumerate(train_loader))
        for i, (ground_truth_image, target) in progress: 
            optimizer.zero_grad()

            sparse_data, context_x, context_y, target_x, target_y = split_target_context(ground_truth_image, image_frame,batch_size, X_SIZE, Y_SIZE, min_context_points, max_context_points)

            pred_y, std_y, log_p, kl, loss_value = model(((context_x, context_y), target_x), target_x.shape[1],
                                            target_y)
            loss_value.backward()
            optimizer.step()

            progress.set_description('E: {} loss: {:.3f}'.format(epoch, loss_value.item()))
        with torch.no_grad():
            # Plot the prediction and the context
            for j, (test_image, test_target) in enumerate(test_loader):
                sparse_data, context_x, context_y, target_x, target_y = split_target_context(test_image, test_image_frame,test_batch_size, X_SIZE, Y_SIZE, min_context_points, max_context_points)


                pred_y, std_y, log_p, kl, loss_value = model(((context_x, context_y), target_x), target_x.shape[1],target_y)
                progress.set_description('E: {} loss: {:.3f}'.format(epoch, loss_value.item()))
                fig, ax = plt.subplots(ncols=2, nrows=2)
                ax[0][1].imshow(test_image.reshape(m,n).cpu(), cmap='gray')
                ax[0][0].imshow(sparse_data.reshape(m,n).cpu(), cmap='gray')
                ax[1][0].imshow(pred_y.detach().reshape(m,n).cpu(), cmap='gray')
                ax[1][1].imshow(std_y.detach().reshape(m,n).cpu(), cmap='gray')
                plt.savefig("attention{}res{}.png".format(epoch, j))
                plt.close()
                if j >= 10:
                    break
                # plot_functions(target_x, target_y, context_x, context_y, pred_y, std_y)
                        # plot_grad_flow(model.named_parameters())

    print("done")

if __name__ == "__main__":
    train_regression()
