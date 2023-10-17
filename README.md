# TextClassification

Implementation of text sentiment analysis

*清华大学2020年人智导大作业（文本情感分类）*

## Directory Tree

```shell
TextClassification/
│  README.md
│  preprocess.ipynb # Preprocessing
|  train.ipynb # training scripts
│  run.py # user interface
|  REPORT.pdf # docs
│
├─.ipynb_checkpoints
│      *.ipynb # jupyter checkpoints
│
├─.vscode
│      settings.json # vscode settings
│
├─data
│      ISEAR ID
│      ... # Train/Test/Validation set
│
├─dataset
│      test_data_x.pt
│      ... # intermediate results (e.g., word table)
│
├─model
│      model.py
|	   cnn.pt 
|  	   ... # model definition
│
├─output
│  └─model
│    ... # TensorBoard outputs
│
└─src
    # other source codes
```

## Getting Started

```shell
# Train the model first
# Then test in the test set with the trainined model
# Pure CNN
python run.py cnn

# CNN + Attention
python run.py cnn_att

# CNN + Inception
python run.py cnn_inception

# RNN
python rnn.py rnn

# FastText
python run.py baseline
```

Note 1：environment：

```python
# OS
Distributor ID: Ubuntu
Description:    Ubuntu 18.04.5 LTS
Release:        18.04
Codename:       bionic
# GPU
NVIDIA-SMI 460.73.01
Driver Version: 460.73.01
CUDA Version: 11.2
GeForce RTX 2080
# Python
Python 3.7.9 (default, Aug 31 2020, 12:42:55) 
[GCC 7.3.0] :: Anaconda, Inc. on linux
# PyTorch
1.7.1
```

Note 2: The tests in this experiment were all run on the GPU. Although the same random seed is set, if the script is run on the CPU, the experimental results may not be fully reproduced (may be slightly different from that in the report). If the script is also run on **GPU**, the experimental results may also not be fully reproduced due to GPU version and other reasons. For discussions in this regard, please refer to the official PyTorch explanation.

Note 3：Please run `run.py` in the GPU environment, because this is a model trained in the GPU environment.
