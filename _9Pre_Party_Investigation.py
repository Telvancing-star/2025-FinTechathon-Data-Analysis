import os
from openai import OpenAI

# 方法1：设置环境变量
os.environ['OPENAI_API_KEY'] = 'sk-803b4d22ada04c749239ed5e2127d205'

client = OpenAI(api_key='sk-803b4d22ada04c749239ed5e2127d205', base_url="https://api.deepseek.com/v1")

response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hello"},
    ],
    stream=False
)

print(response.choices[0].message.content)