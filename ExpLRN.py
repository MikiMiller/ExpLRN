import torch
import numbers
import warnings
import math
from torch.nn.parameter import Parameter
from torch.nn import Module
from torch.nn import functional as F
import torchvision

class FeatExpLRN(Module):
    r"""Similar to  Pytorch local response normalization, this applies our LRN with an exponential kernal
    over an input signal. The shape of the input is (Batch x C x W x H)
    Specifically the feature ExpLRN normalizations across the channel dimension.
    There are four learnable parameters: lambda, alpha, beta, and k

    .. math::
        b_{c}= \frac{a_{c}^{2}}
        {\left(k+ \frac{\alpha \sum_{j=1}^{n} a_{j}^{2} e^{-\left|c-j\right|} }{\sum_{j=1}^{n} e^{-|c-j| / \lambda }} \right)^\beta} 

        = \frac{a_{c}^{2}}{\left(k+ \frac{\alpha \sum_{j=1}^{n} a_{j}^{2} e^{-\left|c-j\right|} }{\lambda } \right)^\beta}

    Args:
        lamb: decay factor for how far out to consider neighbors Default: 2.
        alpha: multiplicative factor. Default: 0.1
        beta: exponent. Default: 1.0
        k: additive factor. Default: 1.

    Shape:
        - Input: :math:`(N, C, *)`
        - Output: :math:`(N, C, *)` (same shape as input)

    Examples::

        >>> lrn = FeatExpLRN(lamb=2.,alpha=.1, beta=1., k=1.)
        >>> output = nn.Conv2d(input)
        >>> output = lrn(output)

    """

    def __init__(self,lamb=2.,alpha=.1, beta=1., k=1.):
        super(LearnLocalResponseNorm, self).__init__()
        #self.neighbors = Parameter(torch.Tensor([neighbors]))
        #self.neighbors = neighbors
        self.lamb = Parameter(torch.Tensor([lamb]))
        self.alpha = Parameter(torch.Tensor([alpha])) 
        self.beta = Parameter(torch.Tensor([beta])) 
        self.k = Parameter(torch.Tensor([k]))

    def forward(self, input):
        return learn_local_response_norm(input,self.lamb,self.alpha, self.beta,
                                     self.k)

    def extra_repr(self):
        #print("self dict",self.__dict__['_parameters'])
        # return 'neighbor={neighbors}, alpha={alpha}, beta={beta}, k={k}'.format(**self.__dict__['_parameters'])
        return 'lambda={lamb},alpha={alpha}, beta={beta}, k={k}'.format(**self.__dict__['_parameters'])


def learn_local_response_norm(input, lamb=5,  alpha=5, beta=0.75, k=1.):
    # type: (Tensor, int, float, float, float) -> Tensor
    """Applies local response normalization over an input signal composed of
    several input planes, where channels occupy the second dimension.
    Applies normalization across channels.
    See :class:`~torch.nn.LocalResponseNorm` for details.
    """
    neighbors=int(torch.ceil(2*4*lamb).item())
    if neighbors%2==0:
        neighbors=neighbors+1
    else:
        pass
    dim = input.dim()
    if dim < 3:
        raise ValueError('Expected 3D or higher dimensionality \
                         input (got {} dimensions)'.format(dim))
    div = input.mul(input).unsqueeze(1)

    sizes = input.size()
    weits = input.clone().detach() 
    # Creates the weights for the exponential kernel 
    weits = weits.new_zeros(([1]+[1]+[int(neighbors)]+[1]+[1]))

    if dim == 3:
        div = F.pad(div, (0, 0, neighbors // 2,  neighbors - 1 // 2))
        div = torch._C._nn.avg_pool2d((div,  neighbors, 1), stride=1).squeeze(1)
    else:
        dev = input.get_device()
        # indexx is a 1D tensor that is a symmetric exponential distribution of some "radius" neighbors
        #idxs = torch.abs(torch.arange(neighbors,device='cuda:%d'%dev)-neighbors//2)
        idxs = torch.abs(torch.arange(neighbors)-neighbors//2)
        weits[0,0,:,0,0]=idxs
        weits= torch.exp(-weits/lamb)
        # creating single dimension at 1;corresponds to number of input channels;only 1 input channel
        # 3D convolution; weits has dims: Cx1xCx1x1 ; this means we have C filters for the C channels
        # The div is the input**2; it has dimensions B x 1 x C x W x H
        div=F.conv3d(div,weits,padding=((neighbors//2),0,0))
        div=div/lamb

    div = div.mul(alpha).add(1).mul(k).pow(beta)
    div = div.squeeze()
    return input.mul(input) / div