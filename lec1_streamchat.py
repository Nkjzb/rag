# Lecture 1
# Building a Streaming Chat Application with OpenAI's ChatGPT
# Requirement
# 1. Support multi-turn conversation
# 2. When the user expresses to you the semantics of wanting 
#    the dialogue to end, please return to the user: 
#    ‘Thank you for your inquiry, goodbye’.
# 3. Support streaming response

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

if __name__ == "__main__":
    chat_with_gpt()
