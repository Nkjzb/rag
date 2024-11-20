import pandas as pd
import numpy as np
from openai import OpenAI
import faiss
import mysql.connector
import json
from typing import List, Tuple
import time
from dotenv import load_dotenv
import os

class TextVectorizer:
    @staticmethod
    def load_config(config_path: str) -> dict:
        """加载配置文件"""
        with open(config_path, 'r') as f:
            return json.load(f)

    def __init__(self, config_path: str):
        """使用配置文件初始化"""
        config = self.load_config(config_path)
        self.client = OpenAI(api_key=config['openai']['api_key'])
        self.mysql_config = config['mysql']
        self.dimension = 1536
        self.index = faiss.IndexFlatL2(self.dimension)
        
    def connect_mysql(self):
        """连接MySQL数据库"""
        return mysql.connector.connect(**self.mysql_config)

    def setup_database(self, schema_file: str):
        """创建数据库表"""
        conn = self.connect_mysql()
        cursor = conn.cursor()
        
        # 先删除已存在的表
        cursor.execute("DROP TABLE IF EXISTS ai_context")
        
        # 创建新表
        with open(schema_file, 'r') as f:
            schema_sql = f.read()
            cursor.execute(schema_sql)
        
        conn.commit()
        cursor.close()
        conn.close()

    def insert_texts_from_file(self, file_path: str) -> List[Tuple[int, str]]:
        """从文本文件读取数据并插入到MySQL"""
        conn = self.connect_mysql()
        cursor = conn.cursor()
        
        inserted_records = []
        
        # 读取文本文件
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                text = line.strip()
                if text:  # 确保不是空行
                    # 插入文本并获取自增ID
                    insert_query = "INSERT INTO ai_context (text) VALUES (%s)"
                    cursor.execute(insert_query, (text,))
                    # 获取插入的ID
                    record_id = cursor.lastrowid
                    inserted_records.append((record_id, text))
        
        conn.commit()
        cursor.close()
        conn.close()
        
        return inserted_records

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """获取文本嵌入向量"""
        try:
            response = self.client.embeddings.create(
                model="text-embedding-ada-002",
                input=texts
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            print(f"Error getting embeddings: {e}")
            return []

    def process_and_store(self, text_records: List[Tuple[int, str]], batch_size: int = 100):
        """处理文本并按MySQL ID顺序存储向量"""
        for i in range(0, len(text_records), batch_size):
            batch = text_records[i:i + batch_size]
            texts = [record[1] for record in batch]  # 获取文本
            
            # 获取embeddings
            embeddings = self.get_embeddings(texts)
            if not embeddings:
                continue

            # 转换为numpy数组
            vectors = np.array(embeddings).astype('float32')
            
            # 添加到FAISS索引
            self.index.add(vectors)
            
            print(f"Processed batch of {len(batch)} items")
            time.sleep(1)  # 避免API限制

    def save_index(self, index_path: str):
        """保存FAISS索引"""
        faiss.write_index(self.index, index_path)

    def load_index(self, index_path: str):
        """加载FAISS索引"""
        self.index = faiss.read_index(index_path)
        
    def search_similar(self, query: str, k: int = 5) -> List[Tuple[int, str, float]]:
        """搜索相似文本"""
        # 获取查询文本的向量
        query_embedding = self.get_embeddings([query])[0]
        query_vector = np.array([query_embedding]).astype('float32')
        
        # 搜索最相似的向量
        distances, indices = self.index.search(query_vector, k)
        
        # 从MySQL获取对应的文本
        conn = self.connect_mysql()
        cursor = conn.cursor()
        
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            # 将numpy.int64转换为Python的int类型
            mysql_id = int(idx) + 1
            cursor.execute("SELECT id, text FROM ai_context WHERE id = %s", (mysql_id,))
            result = cursor.fetchone()
            if result:
                results.append((result[0], result[1], float(distance)))
        
        cursor.close()
        conn.close()
        
        return results

        
# 使用示例
if __name__ == "__main__":
    # 使用配置文件初始化vectorizer
    vectorizer = TextVectorizer('config/config.json')
    
    # 设置数据库
    vectorizer.setup_database("sql/schema.sql")
    # 从文本文件读取并插入数据
    text_records = vectorizer.insert_texts_from_file("data/运动鞋店铺知识库.txt")
    # 处理文本并存储向量
    vectorizer.process_and_store(text_records, batch_size=10)
    
    # 保存FAISS索引
    vectorizer.save_index("data/store_knowledge.index")