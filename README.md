# RGBD-SEG


## Training Data
Downloaded from https://github.com/ankurhanda/nyuv2-meta-data .


## Dependency
- CUDA 8.0
- cuDNN v6
- tensorflow (1.4.0)
- tensorflow-gpu (1.4.0)
- Keras (2.1.1)

## Directory structure
```
├─train.py
├─inference.py
├─nets
│  ├─HogeNet.py
├─modules
|  ├─huga.py
│  └─hoge.py
├─data
   ├─test
   │  ├─color
   │  ├─depth
   │  └─label
   └─train
       ├─color
       ├─depth
       └─label
```