import json

# 读取原始 JSON 文件
with open('ds/alpaca_data_en_52k.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 写入 JSON Lines 文件
with open('ds/alpaca_data_en_52k.jsonl', 'w', encoding='utf-8') as f:
    for item in data:
        f.write(json.dumps(item) + '\n')
