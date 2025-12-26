import os
import requests
import PyPDF2
import json

# 安装PyPDF2：pip install PyPDF2

API_KEY = "sk-803b4d22ada04c749239ed5e2127d205"
url = "https://api.deepseek.com/chat/completions"
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}"
}

# PDF文件目录
pdf_dir = ".\data\Party Profile"

# 结果保存目录
result_dir = ".\output\party_investigation"
os.makedirs(result_dir, exist_ok=True)


# 读取PDF文件内容
def read_pdf(file_path):
    pdf_text = ""
    try:
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                pdf_text += page.extract_text() or ""
        return pdf_text
    except Exception as e:
        print(f"读取PDF文件出错 {file_path}: {e}")
        return ""


# 处理单个政党的PDF文件
def process_party_pdf(pdf_path):
    # 读取PDF内容
    pdf_text = read_pdf(pdf_path)
    if not pdf_text:
        return

    # 获取政党名称（从文件名中提取，去掉.pdf扩展名）
    party_name = os.path.splitext(os.path.basename(pdf_path))[0]

    # 准备API请求数据
    data = {
        "model": "deepseek-chat",
        "messages": [
            {
                "role": "system",
                "content": "选举债券是印度政府在 2018 年推出的一种政治捐款方式，旨在让企业或个人可以向政党匿名捐款。" +
                           "它本质上是一种无记名的银行券，捐赠者在印度国家银行（SBI）购买后，可以把它交给某个政党，政党再兑换成资金。" +
                           "它们可以由个人、团体或企业组织购买并捐赠给他们选择的一方，然后可以在 15 天后无息赎回。" +
                           "你是一位普通的印度选民，在新一轮的选举债券投资中，你需要决定是否购买某个政党的选举债券。" +
                           "具体而言，对每一个政党而言存在让本来就捐过的人继续捐的指标、让本来捐过的人这次不想捐的指标、" +
                           "让本来没有捐过的人这次愿意捐的指标以及让原本未捐者仍然不想捐的指标。" +
                           "维基百科Wikipedia上对各政党的介绍可以提供许多可量化或可比较的线索，我将为你提供各政党的Wikipedia介绍（PDF格式）。" +
                           "你需要根据这些介绍分辨出这些政党在如上四个指标的如下子指标中的表现，输出为布尔值。" +
                           "(1) 让本来就捐过的人继续捐的指标：A1 政绩是否符合承诺、A2 选举表现是否变强、A3 意识形态是否保持一致、A4 内部是否稳定、A5 财务是否透明" +
                           "(2) 让本来捐过的人这次不想捐的指标：B1 是否有腐败或丑闻、B2 是否路线急转、B3 是否出现关键政策失败、B4 是否有候选人争议、B5 选举势头是否下降" +
                           "(3) 让本来没有捐过的人这次愿意捐的指标：C1 政策主张贴是否近其利益或价值观、C2 新领导上台是否有改革潜力、C3 政党势头是否上升、C4 透明度、清廉形象是否优于其他党、C5 地方治理表现是否优秀" +
                           "(4) 让原本未捐者仍然不想捐的指标：D1 党在意识形态上是否远离其立场、D2 选举势头是否下降、D3 是否管理混乱、常有内讧、D4 财务是否透明 D5 是否卷入犯罪、宗派冲突、极端主义争议"
            },
            {
                "role": "user",
                "content": f"政党的Wikipedia PDF文本如下:\n\n{pdf_text[:2000000]}，现在要求对每一个细分指标给出明确的Yes/No的判断。"
            }
        ],
        "stream": False  # 关闭流式传输
    }

    try:
        # 发送API请求
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()  # 检查响应状态

        # 解析API响应
        result = response.json()
        if "choices" in result and len(result["choices"]) > 0:
            party_result = {
                "party_name": party_name,
                "response": result["choices"][0]["message"]["content"],
                "full_response": result
            }

            # 保存结果到JSON文件
            result_file = os.path.join(result_dir, f"{party_name}_result.json")
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(party_result, f, ensure_ascii=False, indent=2)

            print(f"处理完成: {party_name}")
        else:
            print(f"API响应格式错误: {party_name}")

    except requests.exceptions.RequestException as e:
        print(f"API请求出错 {party_name}: {e}")
    except json.JSONDecodeError as e:
        print(f"解析API响应出错 {party_name}: {e}")
    except Exception as e:
        print(f"处理政党出错 {party_name}: {e}")


# 主函数
def main():
    # 获取PDF目录下的所有PDF文件
    pdf_files = [os.path.join(pdf_dir, f) for f in os.listdir(pdf_dir) if f.endswith('.pdf')]

    print(f"找到 {len(pdf_files)} 个PDF文件")

    # 循环处理每个PDF文件
    for pdf_file in pdf_files:
        process_party_pdf(pdf_file)


if __name__ == "__main__":
    main()