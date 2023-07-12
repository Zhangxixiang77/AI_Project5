# 多模态情感分析
The Fifth Project of Contemporary Artificial Intelligence Course, DASE, ECNU


## 环境配置

详见 requirements.txt 使用方式：

```py
pip install -r requirements.txt
```



## 项目结构

```py
|-- dataset  #存放实验数据
    |-- data  #原始数据（仅样例）
    |-- dev.json  #处理后的验证集
    |-- test_without_label.txt  #无标签的测试数据
    |-- test.json  #处理后的测试集
    |-- train.json  #处理后的训练集
|-- model  #模型
    |-- ImageOnly.py  #图像模块
    |-- MultiModal.py  #多模态模型
    |-- TextOnly.py  #文本模块
    |-- utils.py  #工具
|-- data_processing.py  #数据处理
|-- requirements.txt  #环境配置
|-- run.py  #主函数
|-- test_with_label.txt  #预测结果文件
|-- README.md  #该文件
```



## 运行方法

- 配置环境依赖：

```
pip install -r requirements.txt
```

- 数据处理：

```py
python data_processing.py
```

- 训练：

```py
python run.py --do_train
```

- 预测：

```py
python run.py --do_test
```



## 参数选择

```py
--do_train: 用于指定是否进行训练。
--do_test: 用于指定是否进行测试。
--test_output_file: 预测结果输出文件的路径。
--train_file: 训练集文件的路径。
--dev_file: 验证集文件的路径。
--test_file: 测试集文件的路径。
--pretrained_model: 预训练模型的选择。
--lr: 学习率的设置，默认为 1e-5。
--epochs: 迭代次数，默认为 10。
--batch_size: 批处理大小，默认为 4
...
```



## 实验结果

##### 多模态融合模型

| Multimodal_Model                                                    | ACC        |
| :------------------------------------------------------- | :--------- |
| 直接拼接                                 | 0.7140    |
| 多头自注意力                                   | 0.6172    |
| Transformer Encoder | 0.7302 |

##### 消融实验结果

Text_only：0.7098   
Img_only：0.6517 

## 参考
https://arxiv.org/abs/2111.02387    
https://github.com/zdou0830/METER
https://arxiv.org/abs/2201.03545

