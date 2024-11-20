# 运动鞋店铺知识库问答系统

这是一个基于RAG (Retrieval-Augmented Generation) 架构的运动鞋店铺知识库问答系统。系统能够根据用户输入的问题，从知识库中检索相关信息，并生成准确的回答。

## 目录结构

```
├── config
│   └── config.json.template        # 配置文件模板
├── lec1_streamchat.py             # 普通的流式输入输出主程序
├── sql
│   ├── ai_context.sql             # 对话上下文数据库表结构
│   └── schema.sql                 # 数据库主要表结构
└── src
    ├── 01_write_to_faiss_test.py  # Faiss 向量库写入测试
    ├── rag_chat_bot.py            # RAG 问答机器人核心实现
    ├── text_vectorizer.py         # 文本向量化工具类及量化
    └── vectorizer_test.py         # 向量化功能测试
```

## 配置说明

在使用前，需要根据 `config.json.template` 创建 `config.json` 文件，填入相应的配置。


## 使用方法

1. 安装依赖：
```bash
pip install -r requirements.txt
```

2. 启动普通的流式输入输出（第一节课作业）：
```bash
python lec1_streamchat.py
```

3. 完成向量化功能，并测试：
```bash
python src/text_vectorizer.py
python src/vectorizer_test.py
```

4. 启动RAG问答机器人（第二、三节课作业）：
```bash
python src/rag_chat_bot.py
```

## 测试样例

### 向量化功能测试
输出示例：
```
搜索查询: 如何退换货？
----------
ID: 5
相似度距离: 0.2745
文本内容: 退换货政策: 自收到商品12天内，商品未使用且原包装完好可申请退换货，需提供订单号和付款截图。

ID: 8
相似度距离: 0.3190
文本内容: 注意事项: 特价和促销商品不支持退换货，退回商品需妥善包装避免运输损坏。

ID: 6
相似度距离: 0.3234
文本内容: 退款处理: 商品返回检查无误后，退款将在11个工作日内处理完成。
```

### RAG 问答机器人
输入示例：
```
鞋子有什么品牌的呢？
```

输出示例：
```
ChatGPT: 根据提供的参考信息，鞋子的品牌包括Adidas、Nike、Puma、Reebok、New Balance、Asics、Under Armour等知名品牌；另外还有休闲品牌如Converse、Vans，以及高端品牌如Gucci、Balenciaga等。以上品牌均提供各种类型的运动鞋和休闲鞋供选择。

所有鞋款都有正品保证，支持专柜验货。您对哪个系列感兴趣，我可以为您详细介绍。
```

输入示例：
```
好的谢谢 我没有问题了
```

输出示例：
```
感谢您的咨询，再见！
```

## 开发说明

- `text_vectorizer.py`: 负责文本向量化，使用 OpenAI 的 embedding 模型
- `rag_chat_bot.py`: 实现 RAG 架构的问答逻辑
- `sql/`: 包含数据库表结构，用于存储对话历史和上下文
- 测试文件可用于功能验证和调试

## 注意事项

- 请确保 OpenAI API Key 配置正确
- 首次运行需要建立向量库，可能需要一定时间
- 建议使用 Python 3.9 或以上版本