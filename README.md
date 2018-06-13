# Neural-Networks

## Partition function 

<img src="https://latex.codecogs.com/gif.latex?Z&space;=&space;\sum_{j=1}^{2^h}\prod_{i=1}^{v}(e^{-E(v_{i}=1)}&plus;1)" title="Z = \sum_{j=1}^{2^h}\prod_{i=1}^{v}(e^{-E(v_{i}=1)}+1)" />

## Back Propagation

<img src="https://latex.codecogs.com/gif.latex?\frac{\partial&space;E}{\partial&space;w_{ih}}&space;=&space;w_{ho}*(y_j&space;-&space;t_j)*h_{out}*(1-h_{out})*input" title="\frac{\partial E}{\partial w_{ih}} = w_{ho}*(y_j - t_j)*h_{out}*(1-h_{out})*input" />

<img src="https://latex.codecogs.com/gif.latex?\frac{\partial&space;E}{\partial&space;w_{ho}}&space;=&space;(y_j&space;-&space;t_j)*h_{out}" title="\frac{\partial E}{\partial w_{ho}} = (y_j - t_j)*h_{out}" />

## Neural Net

<img src="https://github.com/darc33/Neural-Networks/blob/master/Images/image.png" />

<img src="https://latex.codecogs.com/gif.latex?hidden_{out}=&space;1/(1&plus;e^{-hidden_{in}})" title="hidden_{out}= 1/(1+e^{-hidden_{in}})" />

<img src="https://latex.codecogs.com/gif.latex?C=-\sum&space;tslog(ys)&space;=&space;-[tslog(ys)&plus;\sum_{i\neq&space;s}t_i&space;log(y_i)]" title="C=-\sum tslog(ys) = -[tslog(ys)+\sum_{i\neq s}t_i log(y_i)]" />

<img src="https://latex.codecogs.com/gif.latex?y_s=\frac{e^{z_s}}{\sum_i&space;e^{z_i}}" title="y_s=\frac{e^{z_s}}{\sum_i e^{z_i}}" />

<img src="https://latex.codecogs.com/gif.latex?y_i=\frac{e^{z_i}}{\sum_i&space;e^{z_s}}" title="y_s=\frac{e^{z_s}}{\sum_i e^{z_i}}" />

<img src="https://latex.codecogs.com/gif.latex?z_s&space;=&space;\sum_{n=1}^{n_hid}y_h*w_{hc}" title="z_s = \sum_{n=1}^{n_hid}y_h*w_{hc}" />

<img src="https://latex.codecogs.com/gif.latex?\frac{\partial&space;c}{\partial&space;w_{hc}}&space;=&space;\frac{\partial&space;c}{\partial&space;z_s}&space;*\frac{\partial&space;z_s}{\partial&space;w_c}" title="\frac{\partial c}{\partial w_{hc}} = \frac{\partial c}{\partial z_s} *\frac{\partial z_s}{\partial w_c}" />

<img src="https://latex.codecogs.com/gif.latex?\frac{\partial&space;c}{\partial&space;z_s}&space;=&space;-\left&space;[\frac{\partial&space;c}{\partial&space;y_s}*\frac{\partial&space;y_s}{\partial&space;z_s}&plus;\frac{\partial&space;c}{\partial&space;y_i}*\frac{\partial&space;y_i}{\partial&space;z_s}&space;\right&space;]&space;=&space;-\left&space;[\frac{t_s}{y_s}*\frac{\partial&space;y_s}{\partial&space;z_s}&plus;\frac{t_i}{y_i}*\frac{\partial&space;y_i}{\partial&space;z_s}&space;\right&space;]" title="\frac{\partial c}{\partial z_s} = -\left [\frac{\partial c}{\partial y_s}*\frac{\partial y_s}{\partial z_s}+\frac{\partial c}{\partial y_i}*\frac{\partial y_i}{\partial z_s} \right ] = -\left [\frac{t_s}{y_s}*\frac{\partial y_s}{\partial z_s}+\frac{t_i}{y_i}*\frac{\partial y_i}{\partial z_s} \right ]" />

<img src="https://latex.codecogs.com/gif.latex?\frac{\partial&space;y_s}{\partial&space;z_s}&space;=&space;y_s(1-y_s)" title="\frac{\partial y_s}{\partial z_s} = y_s(1-y_s)" />

<img src="https://latex.codecogs.com/gif.latex?\frac{\partial&space;y_i}{\partial&space;z_s}&space;=&space;-y_s*y_i" title="\frac{\partial y_i}{\partial z_s} = y_s*y_i" />

<img src="https://latex.codecogs.com/gif.latex?\frac{\partial&space;c}{\partial&space;z_s}&space;=&space;-\left&space;[&space;\frac{t_s}{y_s}y_s(1-y_s)&plus;\frac{t_i}{y_i}(-y_sy_i)&space;\right&space;]=-[t_s-y_s(\underset{1}{t_s&plus;t_i})]=y_s-t_s" title="\frac{\partial c}{\partial z_s} = -\left [ \frac{t_s}{y_s}y_s(1-y_s)+\frac{t_i}{y_i}(-y_sy_i) \right ]=-[t_s-y_s(\underset{1}{t_s+t_i})]=y_s-t_s" />

<img src="https://latex.codecogs.com/gif.latex?\frac{\partial&space;z_s}{\partial&space;w_{hc}}&space;=&space;hidden_{out}" title="\frac{\partial z_s}{\partial w_{hc}} = hidden_{out}" />

<img src="https://latex.codecogs.com/gif.latex?{\color{Blue}&space;\frac{\partial&space;c}{\partial&space;w_{hc}}&space;=&space;(y_s-t_s)hidden_{out}}" title="{\color{Blue} \frac{\partial c}{\partial w_{hc}} = (y_s-t_s)hidden_{out}}" />

So the gradient is:

<img src="https://latex.codecogs.com/gif.latex?w_{ho}=class_{loss}&space;&plus;&space;wd_{loss}" title="w_{ho}=class_{loss} + wd_{loss}" />

<img src="https://latex.codecogs.com/gif.latex?w_{ho}=Average\left&space;(&space;\frac{\partial&space;c}{\partial&space;w_{hc}}&space;\right&space;)&space;&plus;&space;\frac{\partial&space;wd_{loss}}{\partial&space;w_{hc}}" title="w_{ho}=Average\left ( \frac{\partial c}{\partial w_{hc}} \right ) + \frac{\partial wd_{loss}}{\partial w_{hc}}" />

<img src="https://latex.codecogs.com/gif.latex?wd_{loss}=\frac{1}{2}\left&space;[&space;\sum&space;\sigma&space;^2&space;\right&space;]&space;*&space;wd_{coefficient}" title="wd_{loss}=\frac{1}{2}\left [ \sum \sigma ^2 \right ] * wd_{coefficient}" />

<img src="https://latex.codecogs.com/gif.latex?\frac{\partial&space;wd_{loss}}{\partial&space;w_{hc}}=&space;w_{hc}\cdot&space;wd_{coeffient}" title="\frac{\partial wd_{loss}}{\partial w_{hc}}= w_{hc}\cdot wd_{coeffient}" />

In the same way:

<img src="https://latex.codecogs.com/gif.latex?\frac{\partial&space;c}{\partial&space;w_{ih}}=&space;(y_s-t_s)*w_{hc}*hidden_{out}*(1-hidden_{out})*x" title="\frac{\partial c}{\partial w_{ih}}= (y_s-t_s)*w_{hc}*hidden_{out}*(1-hidden_{out})*x" />
