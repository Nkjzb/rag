from text_vectorizer import TextVectorizer  # 导入之前实现的向量化类
import json
import os
import time
from openai import OpenAI
from dotenv import load_dotenv
import sys
import readline

# load API key from the env
load_dotenv()

# create OpenAI client and set the API key
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def check_farewell_intent(text):
    """
    使用GPT判断用户输入是否包含结束对话的意图
    """
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "你的任务是判断用户的输入是否表达了想要结束对话的意图。如果是，请只回复'true'，如果不是，请只回复'false'。"},
                {"role": "user", "content": f"用户说：{text}"}
            ],
            temperature=0,  # 使用较低的temperature以获得更确定的答案
            max_tokens=10   # 只需要简短的回复
        )
        result = response.choices[0].message.content.strip().lower()
        return result == 'true'
    except Exception as e:
        print(f"\n判断意图时发生错误: {str(e)}")
        return False

def create_chat_completion(messages, retries=3, delay=2):
    """
    创建聊天完成，包含重试机制
    """
    for attempt in range(retries):
        try:
            return client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                stream=True
            )
        except Exception as e:
            if attempt == retries - 1:  # 最后一次尝试
                raise e
            print(f"\n发生错误，{delay}秒后重试: {str(e)}")
            time.sleep(delay)

def chat_with_gpt():
    # 初始化消息历史，包含系统角色
    messages = [
        {"role": "system", "content": "你是一个有帮助的AI助手，会提供准确、有用的回答。"}
    ]
    
    print("欢迎使用ChatGPT! 输入'quit'退出对话，输入'clear'清除对话历史。")
    
    while True:
        try:
            # 获取用户输入
            user_input = input("\n你: ").strip()
            
            if user_input.lower() == 'quit':
                print("再见！")
                break
            
            if user_input.lower() == 'clear':
                messages = [messages[0]]  # 保留系统消息
                print("对话历史已清除！")
                continue
            
            if not user_input:
                continue
            
            # 检查是否包含告别意图
            if check_farewell_intent(user_input):
                print("ChatGPT: 感谢您的咨询，再见！")
                break
            
            # 将用户输入添加到消息历史
            messages.append({"role": "user", "content": user_input})
            
            # 创建流式响应
            stream = create_chat_completion(messages)
            
            print("ChatGPT: ", end="", flush=True)
            
            # 收集完整响应
            full_response = ""
            
            # 流式输出回复
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    content = chunk.choices[0].delta.content
                    print(content, end="", flush=True)
                    full_response += content
            
            print()  # 换行
            
            # 将助手的回复添加到消息历史
            messages.append({"role": "assistant", "content": full_response})
            
        except KeyboardInterrupt:
            print("\n程序被用户中断")
            sys.exit(0)
        except Exception as e:
            print(f"\n发生错误: {str(e)}")

class RAGChatBot:
    def __init__(self, config_path: str):
        """
        初始化RAG聊天机器人
        :param config_path: 配置文件路径
        """
        # 加载配置
        self.config = self.load_config(config_path)
        
        # 初始化OpenAI客户端
        self.client = OpenAI(api_key=self.config['openai']['api_key'])
        
        # 初始化向量搜索器
        self.vectorizer = TextVectorizer(config_path)
        self.vectorizer.load_index("data/store_knowledge.index")

    @staticmethod
    def load_config(config_path: str) -> dict:
        """加载配置文件"""
        with open(config_path, 'r') as f:
            return json.load(f)

    def get_relevant_context(self, query: str, k: int = 3) -> str:
        """
        获取相关上下文
        :param query: 用户查询
        :param k: 返回的相关文档数量
        :return: 格式化的上下文字符串
        """
        try:
            results = self.vectorizer.search_similar(query, k=k)
            # 格式化上下文
            context_parts = []
            for _, text, score in results:
                context_parts.append(text)
            return "\n".join(context_parts)
        except Exception as e:
            print(f"获取上下文时发生错误: {str(e)}")
            return ""

    def generate_prompt_with_context(self, query: str, context: str) -> str:
        """
        生成包含上下文的prompt
        :param query: 用户查询
        :param context: 相关上下文
        :return: 格式化的prompt
        """
        return f"""基于以下参考信息回答用户的问题。如果参考信息不足以回答问题，请说明无法回答或需要更多信息。

参考信息：
{context}

用户问题：{query}

回答："""

    def chat_with_gpt(self):
        """实现RAG增强的对话功能"""
        messages = [
            {"role": "system", "content": "你是一个知识丰富的AI助手，会基于提供的上下文信息回答问题。如果上下文信息不足，会明确告知用户。"}
        ]
        
        print("欢迎使用RAG增强的ChatGPT! 输入'quit'退出对话，输入'clear'清除对话历史。")
        
        while True:
            try:
                user_input = input("\n你: ").strip()
                
                if user_input.lower() == 'quit':
                    print("再见！")
                    break
                
                if user_input.lower() == 'clear':
                    messages = [messages[0]]
                    print("对话历史已清除！")
                    continue
                
                if not user_input:
                    continue
                
                if check_farewell_intent(user_input):
                    print("ChatGPT: 感谢您的咨询，再见！")
                    break
                
                # 获取相关上下文
                context = self.get_relevant_context(user_input)
                
                # 生成包含上下文的prompt
                prompt_with_context = self.generate_prompt_with_context(user_input, context)
                
                # 将用户输入和上下文添加到消息历史
                messages.append({"role": "user", "content": prompt_with_context})
                
                # 创建流式响应
                stream = create_chat_completion(messages)
                
                print("ChatGPT: ", end="", flush=True)
                
                full_response = ""
                for chunk in stream:
                    if chunk.choices[0].delta.content is not None:
                        content = chunk.choices[0].delta.content
                        print(content, end="", flush=True)
                        full_response += content
                
                print()
                
                messages.append({"role": "assistant", "content": full_response})
                
            except KeyboardInterrupt:
                print("\n程序被用户中断")
                sys.exit(0)
            except Exception as e:
                print(f"\n发生错误: {str(e)}")

if __name__ == "__main__":
    rag_bot = RAGChatBot('config/config.json')
    rag_bot.chat_with_gpt()
