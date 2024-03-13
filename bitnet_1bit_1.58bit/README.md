# See the file READRMB.txt for changes made by RMB.

[![Multi-Modality](agorabanner.png)](https://discord.gg/qUtxnK2NMf)

# BitNet
![bitnet](/bitnet.png)
PyTorch Implementation of the linear methods and model from the paper "BitNet: Scaling 1-bit Transformers for Large Language Models"

[Paper link:](https://arxiv.org/pdf/2310.11453.pdf)

BitLinear = tensor -> layernorm -> Binarize -> abs max quantization -> dequant

"The implementation of the BitNet architecture is quite simple, requiring only the replacement of linear projections (i.e., nn.Linear in PyTorch) in the Transformer. " -- BitNet is really easy to implement just swap out the linears with the BitLinear modules! 

## **NEWS**
- BitNet Transformer has been trained using the `train.py` file that trains on enwiki8 a small 1gb dataset of wikipedia: [HERE IS THE LINK](https://drive.google.com/file/d/1gBuZRFBqMV3cVD902LXA_hmZl4e0dLyY/view)
- **New Iteration** 🔥 There is an all-new iteration from the paper "[The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits](https://arxiv.org/abs/2402.17764)", we're implementing it now. Join the Agora discord and contribute! [Join Here](https://discord.gg/hFzevCjG8c)
- **New Optimizations** The first `BitLinear` has been optimized and we now have a Bit Attention `BitMGQA` That implements BitLinear into the attention mechanism. Multi Grouped Query Attention is also widely recognized as the best attention for its fast decoding and long context handling, thanks to Frank for his easy to use implementation!

## Appreciation
- Dimitry, Nullonix for analysis and code review and revision
- Vyom, for providing 4080 to train!

## Installation
`pip install bitnet`

## Usage:

### `BitLinear`
- Example of the BitLinear layer which is the main innovation of the paper!
```python
import torch

from bitnet import BitLinear

# Input
x = torch.randn(10, 512)

# BitLinear layer
layer = BitLinear(512, 400)

# Output
y = layer(x)

print(y)
```
----

### `BitNetTransformer`
- Fully implemented Transformer as described in the diagram with MHA, and BitFeedforwards
- Can be utilized not just for text but for images and maybe even video or audio processing
- Complete with residuals and skip connections for gradient flow

```python
import torch

from bitnet import BitNetTransformer

bitnet = BitNetTransformer(
    num_tokens=20000,
    dim=512,
    depth=6,
    dim_head=64,
    heads=8,
    ff_mult=4,
)

tokens = torch.randint(0, 20000, (1, 512))
logits = bitnet(tokens)
print(logits.shape)
```


### `BitAttention`
This Attention has been modified to use BitLinear instead of the default linear projection. It's also using Multi-Grouped Query Attention instead of regular multi-head attention for faster decoding and longer context handling.

```python
import torch
from bitnet import BitMGQA

# Create a random tensor of shape (1, 10, 512)
x = torch.randn(1, 10, 512)

# Create an instance of the BitMGQA model with input size 512, 8 attention heads, and 4 layers
gqa = BitMGQA(512, 8, 4)

# Pass the input tensor through the BitMGQA model and get the output and attention weights
out, _ = gqa(x, x, x, need_weights=True)

# Print the shapes of the output tensor and attention tensor
print(out)
```

### `BitFeedForward`
- Feedforward as shown in the diagram with BitLinear and a GELU:
- Linear -> GELU -> Linear
- You can add dropouts, or layernorms, or other layers for a better ffn

```python
import torch
from bitnet import BitFeedForward

# Create a random input tensor of shape (10, 512)
x = torch.randn(10, 512)

# Create an instance of the BitFeedForward class with the following parameters:
# - input_dim: 512
# - hidden_dim: 512
# - num_layers: 4
# - swish: True (use Swish activation function)
# - post_act_ln: True (apply Layer Normalization after each activation)
# - dropout: 0.1 (apply dropout with a probability of 0.1)
ff = BitFeedForward(512, 512, 4, swish=True, post_act_ln=True, dropout=0.1)

# Apply the BitFeedForward network to the input tensor x
y = ff(x)

# Print the shape of the output tensor y
print(y)  # torch.Size([10, 512])
```

## Inference
```python
from bitnet import BitNetInference

bitnet = BitNetInference()
bitnet.load_model("../model_checkpoint.pth")  # Download model
output_str = bitnet.generate("The dog jumped over the ", 512)
print(output_str)
```

## Huggingface Usage
```python
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from bitnet import replace_linears_in_hf

# Load a model from Hugging Face's Transformers
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Replace Linear layers with BitLinear
replace_linears_in_hf(model)

# Example text to classify
text = "Replace this with your text"
inputs = tokenizer(
    text, return_tensors="pt", padding=True, truncation=True, max_length=512
)

# Perform inference
model.eval()  # Set the model to evaluation mode
with torch.no_grad():
    outputs = model(**inputs)
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    print(predictions)

# Process predictions
predicted_class_id = predictions.argmax().item()
print(f"Predicted class ID: {predicted_class_id}")

# Optionally, map the predicted class ID to a label, if you know the classification labels
# labels = ["Label 1", "Label 2", ...]  # Define your labels corresponding to the model's classes
# print(f"Predicted label: {labels[predicted_class_id]}")
```

# License
MIT

# Citation
```bibtex
@misc{2310.11453,
Author = {Hongyu Wang and Shuming Ma and Li Dong and Shaohan Huang and Huaijie Wang and Lingxiao Ma and Fan Yang and Ruiping Wang and Yi Wu and Furu Wei},
Title = {BitNet: Scaling 1-bit Transformers for Large Language Models},
Year = {2023},
Eprint = {arXiv:2310.11453},
}

```


# Todo
- [x] Double check BitLinear implementation and make sure it works exactly as in paper 
- [x] Implement training script for `BitNetTransformer`
- [x] Train on Enwiki8, copy and past code and data from Lucidrains repos
- [x] Benchmark performance
- [x] Look into Straight Through Estimator for non-differentiable backprop
- [x] Implement BitFeedForward
- [x] Clean up codebase 
- [x] Add unit tests for each module
- [ ] Implement the new BitNet1.5b from the [paper](https://arxiv.org/abs/2402.17764)
- [ ] Implement the BitNet15b in Cuda
