# -*- coding: utf-8 -*-
"""
Created on Mon Mar  3 09:06:05 2025

@author: TechnoLEDs
https://pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html
"""

import torch
import numpy as np

data = [[1, 2],[3, 4]]
print('data:', data)
x_data = torch.tensor(data)
print('\ntensor:\n', x_data)
np_array = np.array(data)
print('\nnp array:\n', np_array)
x_np = torch.from_numpy(np_array)
print('\ntensor form np array:\n', x_np)
np_x = x_np.numpy()
print('\nnp array from tensor:\n', np_x)

x_ones = torch.ones_like(x_data) # retains the properties of x_data
print(f"\nOnes Tensor: \n {x_ones} \n")

x_rand = torch.rand_like(x_data, dtype=torch.float) # overrides the datatype of x_data
print(f"Random Tensor: \n {x_rand} \n")

shape = (2,3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")

tensor = torch.rand(3,4)

print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")

# We move our tensor to the current accelerator if available
if torch.accelerator.is_available():
    tensor = tensor.to(torch.accelerator.current_accelerator())
print(f"Device tensor is stored on: {tensor.device}")
    
tensor = torch.ones(4, 4)
print(f"First row: {tensor[0]}")
print(f"First column: {tensor[:, 0]}")
print(f"Last column: {tensor[..., -1]}")

tensor[:,1] = 0
print(tensor)
t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(t1)

a = torch.tensor([[1,1],[2,2]], dtype=float)
print(a)
b = a.T
print(b)

# This computes the matrix multiplication between two tensors. 
y1 = a @ b
print(y1)

# This computes the element-wise product. 
z1 = a * b
print(z1)
z2 = a.mul(b)
print(z2)

y2 = torch.rand_like(a)
print(y2)
torch.mul(a, b, out=y2)
print(y2)

y3 = a.matmul(b)
print(y3)

y4 = torch.rand_like(a)
print(y4)
y5 = torch.matmul(a, b, out=y4)
print(y5)


agg = a.sum()
agg_item = agg.item()
print(agg_item, type(agg_item))

print(f"{a} \n")
# copy a tensor and detach from the computation graph
c = a.clone().detach()
print(c)
c.add_(5)
print(c)

x = torch.zeros(2, 1, 2, 1, 2).int()
x.size()
# Returns a tensor with all specified dimensions of input of size 1 removed
y = torch.squeeze(x)
y.size()
# Returns a new tensor with a dimension of size one inserted at the specified position.
y1 = torch.unsqueeze(y, 0)
y1.size()
y1.type()

y2 = y.unsqueeze(1)
y2.shape

y3 = torch.unsqueeze(y, 2).float()
y3.size()
y3.type()

# squeeze(input, 0) leaves the tensor unchanged
y = torch.squeeze(x, 0)
y.size()
# squeeze(input, n) will squeeze size 1 of tensor on first 'n' dimension shape 
y = torch.squeeze(x, 1)
y.size()
z = torch.squeeze(y, 2)
z.size()
y = torch.squeeze(x, (1, 2, 3))
y.size()

# np array
s = np.expand_dims(x, axis=0) 
s.shape
s = np.expand_dims(x, axis=-1) 
s.shape

# Returns a view of the original tensor input with its dimensions permuted.
x1 = torch.zeros(1, 2, 3, 4)
x1.size()
s = torch.permute(x1, (3, 2, 0, 1))
s.shape

