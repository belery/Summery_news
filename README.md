# 新闻摘要项目

使用cnews数据集和transformer模型实现新闻内容摘要功能

## 项目结构

```
.
├── README.md
├── requirements.txt
└── src
    ├── main.py              # 主程序入口
    ├── data                 # 数据处理模块
    │   ├── __init__.py
    │   └── cnews_data_processor.py
    ├── model                # 模型定义和训练模块
    │   ├── __init__.py
    │   ├── transformer_summarizer.py
    │   └── trainer.py
    └── utils                # 工具模块
        ├── __init__.py
        └── helpers.py
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 训练模型

```bash
python src/main.py --data_path /path/to/cnews/data --mode train --epochs 10 --batch_size 32 --learning_rate 0.001
```

### 生成摘要

```bash
python src/main.py --data_path /path/to/cnews/data --mode predict --model_path /path/to/model
```

## 模块说明

### 数据处理模块 (src/data)

- `CNewsDataProcessor`: 处理CNews数据集，包括加载、预处理和分词功能

### 模型模块 (src/model)

- `TransformerSummarizer`: 基于Transformer的摘要生成模型
- `ModelTrainer`: 模型训练和评估工具

### 工具模块 (src/utils)

- `helpers.py`: 辅助函数，如掩码创建、分数计算等

### 主程序 (src/main.py)

- 提供训练和预测两种模式的命令行接口