# SST_Flow_Recognition
基于EEG信号时域 空域 频域三种特征的心流识别
> keywords: attention, DenseNet, CNN, EEG, Emotion recognization, Spectral-Spatial-Temporal, Flow
### 项目文件及文件夹介绍
* **config:** 存放配置文件
* **data_preprocess:** 负责数据的初始标准化 数据封装打包为npy文件 到database目录
* **data_spectral/data_temporal:** 数据时域数据以及频域微分熵数据
* **database:** 打包好的数据目录
* **model:** SST_model
* **run.py/run_cpu.py/run_gpu.py:** 不同环境下运行文件
* **model_auth.png:** 模型架构
### 项目运行
#### 运行环境
#### 运行命令
```python run.py -c ./config/SEED.ini```
