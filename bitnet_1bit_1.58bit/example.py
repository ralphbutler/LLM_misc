import torch

## from bitnet import BitLinear  ## RMB  (either this one ...
## from bitnet.bitlinear import BitLinear  # RMB  ... or this one)

from bitnet.bitbnet_b158 import BitLinear15b as BitLinear

# Input
x = torch.randn(10, 512)

# BitLinear layer
layer = BitLinear(512, 400, bias=None)

# Output
y = layer(x)

print(y)
