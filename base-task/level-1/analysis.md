# Architecture Analysis

## Residue & Skip Connection
- I picture the residue and the skip connection block to be a mechanism which allows the model to 'enhance' it's previous understanding rather than completely transforming it.  It is like building on top of what is already known rather than using a fresh tool to interpret what is already known.
- The residue here is the transformed form of what is already known, the enhancement, and the skip connection is what merges it with the plain untransformed form. This ensures we get the best of both.
- Skip connection branch greatly mitigates the problem of vanishing gradients as it adds the raw values from the previous layers ensuring that, even if the residue vanishes, the final sum sustains and remains finite and stable. 
- This mechanism is the reason, why gradients flow with ease as there is basically a direct connection that exists with no big changes from the input to the output of the residual block. 

## Network Depth
- Usually, when we train a dataset on models with increasing complexity, there is a possibility that, the model with a deeper architecture may undergo degradation, meaning it halts learning and navigating the loss function in the postive direction. Instead it may underperform
- Residues and Skip connections help prevent this by allowing each block to refine the input representation instead of learning an entirely new mapping, making optimization considerably easier.
- Also, deeper models are prone to overfitting very easily 

## Challenges
- The biggest challenge was hyperparameter tuning
- A 89% accuracy was obtained relatively easily
- The difficulty came with squeezing the accuracy up to 95%
- To achieve this I had to try out a variety of optimizers, schedulers, etc
- During initial training, I faced slight overfitting which was clearly indicated by the gap between the train and validation loss. This was fixed by introducing dropouts, regularisation and label smoothing.
- Also, managing dimensions and keeping track of tensors was a bit difficult in the start, but was very manageable