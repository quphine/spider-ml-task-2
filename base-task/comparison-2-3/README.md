# Comparison

> The file LSTM-reimplementation.ipynb uses a predefined LSTM block, the original vanilla implementation (from scratch, part of level-2) is in the /base-task/level-2 directory

The transformer and the LSTM architectures were implemented with two different input and output sequence lengths initially, as a part of levels 2 and 3. To ensure fair comparison between the architectures for the Jena Climate dataset, the LSTM was re-implemented under the file "LSTM-Transformer.ipynb" with changed sequence lengths, in this directory.

The following results were obtained. 

## LSTM vs Transformer Performance 

The LSTM consistently outperformed the Transformer across every evaluation metric. While the difference may appear numerically small, the improvements are systematic and indicate that the LSTM is better suited to the Jena Climate forecasting task under the current experimental setup.

| Metric | LSTM | Transformer | Better |
|--------|------:|------------:|:------|
| MAE | **1.756 °C** | 1.825 °C | LSTM ↓ 3.8% |
| RMSE | **2.318 °C** | 2.417 °C | LSTM ↓ 4.1% |
| R² Score | **0.9001** | 0.8914 | LSTM |
| Correlation | **0.9490** | 0.9446 | LSTM |
| Best MAE | **0.322 °C** | 0.382 °C | LSTM |
| Median MAE | 1.614 °C | **1.597 °C** | Transformer (slightly) |
| Worst MAE | **6.326 °C** | 7.228 °C | LSTM |

## Error Distribution

The LSTM produced fewer large forecasting errors across all error thresholds.

| Error Threshold | LSTM | Transformer |
|----------------|-----:|------------:|
| > 1°C | **1272** | 1305 |
| > 2°C | **491** | 514 |
| > 3°C | **122** | 175 |
| > 4°C | **35** | 49 |
| > 5°C | **11** | 13 |
| > 6°C | **3** | 4 |

The gap becomes increasingly noticeable for larger errors. Above the 3°C threshold, the Transformer produces approximately **43% more** high-error samples (175 vs 122), suggesting that its prediction failures are more severe when it makes mistakes.

## Possible explanation
In this section, I attempt to explain, why the transformer failed to beahve up to the mark

- Dataset Size - Transformers are known to require extensive training on notoriously big datasets for better outputs
- CNN Downsampling - Unlike the LSTM, the transformer was fed input from after downsampling it with a `conv1D` layer due to computational cost/overhead
- Optimization Difficulty - The transformer might have needed more attendive hyperparameter tuning

## Training stability
- The LSTM had an extremely stable training (After correctly initializing the weights with Xavier and Orthogonal inits as recommended by the paper)
- The Transformer was extremely unstable initially, with highly fluctuating loss values. But, when the post-layer norm was replaced by pre-layer norm, the training stabilized greatly.
- Also, the transformer was overfitting to a great extend. After constant changes to the architecuture was when I actually got a stable training curve, which lasted for a few epochs, followed by aggressive overfitting.

## Runtime and Memory 
- The transformer was computationally cumbersome. Despite reduction of the input dim by a convolution layer the O($n^{2}$) complexity was huge for the GPUs to handle
- The LSTM had a greater training time. Because of its recursive nature, the model training took extra time on comparison to the transformer, which trained faster.
- In terms of inference, both were equally fast/ the difference is insignificant (possibly due to input size/computational capabilities)

## Conclusion 
Attention enables Transformers to better model long-range dependencies by letting every token directly interact with every other token and enabling highly parallel computation. Conversely, LSTMs process sequences sequentially through recurrent hidden states, which makes them slower to train on long sequences and more prone to lose information over time. However, LSTMs still are a good choice for smaller data sets, resource-constrained environments and streaming or real-time applications where low memory usage and sequential processing are helpful. Transformers tend to do better than LSTMs on larger data sets and tasks with longer sequences, where the ability to model global context outweighs the higher computational and memory costs.
