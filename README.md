# An Efficient Frequency Domain Separation Network for Paired and Unpaired Image Super-Resolution
This is an official pytorch implementation of the paper: "An Efficient Frequency Domain Separation Network for Paired and Unpaired Image Super-Resolution"

## Abstract
Although existing super-resolution (SR) techniques have made great progress, they are often tailored for either paired or unpaired scenery, thus may result in poor migration ability. In this work, we propose a generalized Frequency Domain Separation Network (FDSNet) for both paired and unpaired SR settings. Firstly, through statistical analysis, we found that real-world low-resolution (LR) images and high-resolution (HR) images differ greatly in high frequencies but less in low frequencies. Inspired by this, we perform high and low-frequency separation of LR images and guide our model to reconstruct the HR contents in the different frequency domains. Then, according to the varying attention on frequencies of traditional CNN and Transformer models, we design a parallel pipeline: LFNet based on Transformer for low-frequency feature extraction, and HFNet based on CNN for high frequencies. In LFNet, to further alleviate the high complexity and data dependency of Transformer, Simplified Multi-head Self Attention (SMSA) is proposed at a low computational cost. And original MLP is replaced by our Spatial Enhancement MLP (SEMLP) to take full advantage of local spatial contexts. Finally, to further facilitate frequency separation and learning, a Frequency attention block is designed to impose guidance on high frequencies. Experiments indicate that our FDSNet achieves promising performance in terms of quantitative and qualitative evaluations while enjoying a faster speed and much fewer parameters.

## FDSNet
![Image text](https://github.com/HappinessL/FDSNet/blob/main/FDSNet.png)
## TransBlock
![Image text](https://github.com/HappinessL/FDSNet/blob/main/Transformer.png)
