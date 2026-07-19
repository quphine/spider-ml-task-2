# LSTM - Architecture analysis
LSTM comprises of 3 gates, as mentioned below:
- Forget Gate
- Input Gate
- Output Gate

Consider the 14 elemental tensor (for one timestamp) entering the LSTM block. Note that before passing this tensor into the model, it is always uupsampled to (B, 128), seperately for each gates. 

## Forget Gate 

- The input tensor is transformed to (B, 128)
- The hidden state of the previous block is identity transformed to (B, 128)
- These tensors are then added, summed up with a bias term
- Squeezed with a sigmoid to yield a value between [0,1]
- This is used as the retention factor (This value is multiplied with the tensor of the Long term memory lane)

So if this value obtained is 0.5, it is analogous to saying, remember 50% of the present long term memory and forget the rest. 
This can be mathematcially stated as,

$$ft​=σ(W_f​x_t​+U_f​h_{t−1}​+b_f​)$$

## Input Gate

- The same steps as follwed in the forget gate are adopted till the sigmoid.
- Once again the process is repeated with freshly transformed inputs and the squeeze function is now replaced by $\tanh(x)$ to introduce negative values
- The activations produced by sigmoid and tanh are then multiplied to obtain the final tensor of dim (B, 128)
- Finally, this tensor is added to the long term memory
This is mathematically stated as,
$$i_t​=σ(W_i​x_t​+U_i​h_{t−1}​+b_i​)$$
$$\tilde{c}_t​=\tanh(W_c​x_t​+U_c​h_{t−1}​+b_c​)$$
$$c_t​=f_t​⊙c_{t−1}​+i_t​⊙\tilde{c}_t​$$

##  Output Gate

- The same process in the forget gate is repeated till the sigmoid activations are obtained.
- Following this, the LTM (long term memory) is squeezed with tha tanh activation, which is then multiplied with the previously obtained activations
- The obtained final tensor (B, 128) is the output or the next hidden state (Short term memory)

$$o_t​=σ(W_o​x_t​+U_o​h_{t−1}​+b_o​)$$
$$h_t​=o_t​⊙tanh(c_t​)$$

## Aftermath/Post Processing
The above sequence is repeated n times depending on the size of the input sequence. After that, we have the final (B, 128) tensor, which is decoded by a simple MLP, that outputs the predictions of the required output length. It is important to note that the methodology equipped in the following process is non-auto-regressive, meaning, we do not call the LSTM blocks recursively to build on it's own predictions to obtain the output sequence of the required length.


## Training Stability

The training process was initially unsatable with oscillating/nan valued val and train losses, this was fixed with proper Xavier initializations for the input to hidden weights and orthogonal initialisations for the hidden to hidden weights. Also, biases were initialized as 1. These changes had an almost immediate effect on training stability.

## Sequence Length and forecasting Challenges

The LSTM architecture was implemented with two different sequence sizes. 432, and 4320 timestamps. It was clearly visible that the LSTM performed significantly better in the case when it had fewer time stamps to process (0.0469 MSE vs 0.10 MSE). Also the main forecasting challenge is error accumulation. Despite evading autoregression, we still are faced with error accumulation. The one shot prediction does grow increasingly deviant from the ground truth for greater distant time stamps in the prediction horizon. 