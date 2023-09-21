# ALAM: Averaging for Low-Precision Activation for memory efficient training

This repository is the official implementation of https://openreview.net/forum?id=BG7H1XsMG0.

+ We propose the Average Quantization algorithm, which generates high-quality sub-1b compressed activations, enabling the allocation of more bits to high-sensitivity activations.
+ We propose the GradNorm Variance algorithm that calculates sensitivity from the variance of L2 normalization values of parameter gradients, eliminating the necessity to store all parameter gradients from multiple seeds.
+ By adopting Average Quantization and GradNorm Variance, we propose ALAM, which achieves a much better trade-off between accuracy and memory savings compared with GACT. Our ALAM can compress activations by up to 12.5x in LLaMA-7B.

## Abstract
 During the training of large language models, the memory required for activation occupies a substantial portion, which further increases as batch size or sequence length expands. Activation Compression Training (ACT) frameworks have been proposed to reduce this activation memory, however, these frameworks prove challenging to implement in sensitive transformer models, and even when applied, they often lead to considerable reductions in accuracy due to the inferior quality of the compressed activations at extremely low bit. In this paper, we introduce ALAM, a novel ACT framework that utilizes average quantization and gradient normalization variance. First, we demonstrate that when compressing activation to group average, the gradient variance can be minimized. Building on this, we propose Average Quantization, which allows us to allocate fewer than 1-bit to activations with low sensitivity, thereby providing room to assign more bits to activations with high sensitivity. Secondly, in contrast to the previous activation compression training necessitates all parameter gradients from various seeds for sensitivity, we present the GradNorm Variance algorithm, which solely relies on L2-normalization value of parameter gradients for determining sensitivity, thereby substantially reducing the memory overhead associated with calculating sensitivity. The ALAM successfully reduces activation memory significantly without compromising accuracy across Transformer models of varying sizes. Notably, the ALAM has attained up to a 12.5x compression rate in Large Language Models, such as LaMMA-7B.
![overview_final](https://github.com/KH9NHAKRFF/ALAM/assets/144604248/fe409dda-cfe1-4b1e-84cb-5425e0aaa5a5)




## Install

```bash
# install pytorch. we use torch 1.10.1 and torch 2.0.1, but other version is also possible 
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia

# build ALAM
pip install -v -e .
```

## Usage 

```python
import alam 
from alam.controller import Controller # import alam controller
# set the target bit and total_step during training. 
# alam will calculate the sensitivity at the initial and 10% of the total step. 
alam.set_config(bit=bit, total_step=total_step)  
model = .... # define your model here
controller = Controller(model)
def pack_hook(tensor): # quantize hook
    return controller.quantize(tensor)
        
def unpack_hook(tensor): # dequantize hook
    return controller.dequantize(tensor)

controller.install_hook()

with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
 for epoch in ...
   for iter in ....
     ......
     def backprop():
         model.train() # make sure you are in the training mode
         output = model(input) # forward
         loss = calculate_loss()
         optimizer.zero_grad() # this line must be present!
         loss.backward() # backward

     optimizer.step() # update the parameters
     controller.iterate(backprop) # tell gact how to perform forward/backward

controller.uninstall_hook()
```
## Results

### 1. Text classification
![results_text_classification](https://github.com/KH9NHAKRFF/ALAM/assets/144604248/4a91560a-753d-49cd-a0af-52767ec2f790)


### 2. Large Language Models
![results_llm](https://github.com/KH9NHAKRFF/ALAM/assets/144604248/d552cc1e-5d86-4ae7-b06e-531b385c079a)


## Example
[text_classification](https://github.com/KH9NHAKRFF/ALAM/tree/main/benchmark/text_classification)

[llm](https://github.com/KH9NHAKRFF/ALAM/tree/main/benchmark/llm)

 
## Acknowledgments
  
  In this repository, [GACT](https://github.com/LiuXiaoxuanPKU/GACT-ICML) are modified to develop our ALAM.
  Thanks the authors for open-source code.
  
 ## Lisense

> All content in this repository is licensed under the MIT license. 

