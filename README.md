# Neural-Networks

## Partition function 

<img src="https://latex.codecogs.com/gif.latex?Z&space;=&space;\sum_{j=1}^{2^h}\prod_{i=1}^{v}(e^{-E(v_{i}=1)}&plus;1)" title="Z = \sum_{j=1}^{2^h}\prod_{i=1}^{v}(e^{-E(v_{i}=1)}+1)" />

## Back Propagation

<img src="https://latex.codecogs.com/gif.latex?\frac{\partial&space;E}{\partial&space;w_{ih}}&space;=&space;w_{ho}*(y_j&space;-&space;t_j)*h_{out}*(1-h_{out})*input" title="\frac{\partial E}{\partial w_{ih}} = w_{ho}*(y_j - t_j)*h_{out}*(1-h_{out})*input" />

<img src="https://latex.codecogs.com/gif.latex?\frac{\partial&space;E}{\partial&space;w_{ho}}&space;=&space;(y_j&space;-&space;t_j)*h_{out}" title="\frac{\partial E}{\partial w_{ho}} = (y_j - t_j)*h_{out}" />

