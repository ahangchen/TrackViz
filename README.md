# TFusion

## 代码结构
 - ctrl: 融合调度层
 - data: 数据配置目录, 存储图像数据与预测数据
 - feature: 提取时空特征
 - post_process: 测试与评估，可视化之前的后处理
 - profile: 初始参数配置
 - pre_process: 预处理层
 - train: 时空模型构建，核心的融合分类器
 - util: 文件，字符串处理等工具
 - viz: 可视化

## 功能信息
  - 全局调度入口：[ctrl/transfer.py](ctrl/transfer.py)
    - 调用rank-reid工程代码，进行图像相似度计算，
    - 计算图像预测准确率
    - 利用图像数据建立时空模型
    - 融合分类器预测
    - 计算融合预测结果准确率
    - market的python测试结果只有rank1-acc和map，rank5和rank10需要用matlab代码计算
  - 各种可视化： [delta_track.py](viz/delta_track.py)

##　Citation

Please cite this paper in your publications if it helps your research:

@article{,
  title={Unsupervised Cross-dataset Person Re-identification by Transfer Learning of Spatial-Temporal Patterns},
  author={Jianming, Lv and Weihang, Chen and Qing, Li and Can, Yang},
  journal={arxiv},
  year={2017}
}
                      
