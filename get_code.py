import yaml
import re
from openai import OpenAI

# -----------------------------------------------------------------------------
# 1. 配置与加载
# -----------------------------------------------------------------------------
def load_prompt_config(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

# 加载你的 prompt-EN.yaml
config = load_prompt_config('prompt-EN.yaml')

client = OpenAI(
    base_url='https://api-inference.modelscope.cn/v1',
    api_key='ms-f4b17c5c-30f3-488d-b87c-27da987fce1d', # 你的 Token
)

extra_body = {
    "enable_thinking": True,
}

messages = [
    {'role': 'system', 'content': config['system']},
    {'role': 'user', 'content': config['user_template']}
]

# -----------------------------------------------------------------------------
# 2. 发送请求
# -----------------------------------------------------------------------------
print(">>> Sending request to ModelScope...")
response = client.chat.completions.create(
    model='Qwen/Qwen3-32B',
    messages=messages,
    stream=True,
    extra_body=extra_body
)

# -----------------------------------------------------------------------------
# 3. 处理流式响应 & 拼接代码
# -----------------------------------------------------------------------------
print("\n=== Thinking Process (CoT) ===\n")

full_code_content = ""  # 用于存储完整的代码字符串
done_thinking = False

for chunk in response:
    if chunk.choices:
        delta = chunk.choices[0].delta
        
        # 获取思考过程
        thinking_chunk = getattr(delta, 'reasoning_content', '') or ''
        # 获取正式回答 (代码部分)
        answer_chunk = getattr(delta, 'content', '') or ''
        
        if thinking_chunk:
            print(thinking_chunk, end='', flush=True)
            
        elif answer_chunk:
            if not done_thinking:
                print('\n\n=== Final Python Code ===\n')
                done_thinking = True
            
            # 1. 实时打印到屏幕
            print(answer_chunk, end='', flush=True)
            
            # 2. 实时拼接到变量中 (用于保存)
            full_code_content += answer_chunk

# -----------------------------------------------------------------------------
# 4. 清洗数据并保存为 .py 文件
# -----------------------------------------------------------------------------
def extract_python_code(text):
    """
    去除 Markdown 的 ```python 和 ``` 标记，只保留代码内容。
    """
    # 正则匹配 ```python ... ``` 中间的内容
    pattern = r"```python\n(.*?)```"
    match = re.search(pattern, text, re.DOTALL)
    
    if match:
        return match.group(1)
    
    # 如果没找到标记，尝试找 ``` ... ```
    pattern_generic = r"```\n(.*?)```"
    match_generic = re.search(pattern_generic, text, re.DOTALL)
    if match_generic:
        return match_generic.group(1)
        
    # 如果完全没有 markdown 标记，则假设全部都是代码
    return text

print("\n\n>>> Saving to file...")

# 清洗代码
clean_code = extract_python_code(full_code_content)

# 保存文件
output_filename = "generated_task.py"
with open(output_filename, "w", encoding="utf-8") as f:
    f.write(clean_code)

print(f">>> Success! Code saved to '{output_filename}'")