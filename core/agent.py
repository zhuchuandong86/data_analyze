import os
import sys
import io
import re
import glob
from contextlib import redirect_stdout
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from utils.reporter import generate_html_report

def run_agent_pipeline(df: pd.DataFrame, user_query: str, api_key: str, api_base: str):
    report_out = "final_report.html"

    # ==========================================
    # 【数据预处理】清理表头
    # ==========================================
    df.columns = df.columns.str.strip().str.replace('\n', '').str.replace('\r', '')
    columns_str = ", ".join(df.columns)
    sample_str = df.head(3).to_string()  

    # ==========================================
    # 🌟 异构模型配置
    # ==========================================
    llm_coder = ChatOpenAI(temperature=0, model="qwen2.5-coder-32b-instruct", api_key=api_key, base_url=api_base, timeout=120)
    llm_planner = ChatOpenAI(temperature=0, model="deepseek-v3-0324", api_key=api_key, base_url=api_base, timeout=60)
    llm_analyst = ChatOpenAI(temperature=0.3, model="deepseek-v3-0324", api_key=api_key, base_url=api_base, timeout=120)
    llm_judge = ChatOpenAI(temperature=0, model="deepseek-v3-0324", api_key=api_key, base_url=api_base, timeout=120)

    # ==========================================
    # Node 0：🗺️ Planner Agent — 动态制定分析计划
    # ==========================================
    print("\n===========================================")
    print("[*] 🗺️ Planner Agent 正在分析数据，制定分析计划...")
    print("===========================================\n")

    planner_prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一个数据分析规划师。根据用户提供的数据表信息，制定一份简洁的分析计划。
【输出格式要求】严格按如下格式输出，不要多余废话：
数据类型: <如：电信网络数据/财务数据>
分析维度:
1. <维度一>
2. <维度二>
核心指标: <列出关键字段名>
图表建议: <建议生成哪几类图表>"""),
        ("user", "表头: {columns}\n\n数据样本:\n{sample}\n\n需求: {query}")
    ])

    analysis_plan = ""
    try:
        for chunk in (planner_prompt | llm_planner).stream({"columns": columns_str, "sample": sample_str, "query": user_query}):
            analysis_plan += chunk.content
            print(chunk.content, end="", flush=True)
    except Exception as e:
        analysis_plan = "通用数据分析：趋势与对比"

    # ==========================================
    # Node 1 & 2：👨‍💻 程序员 Agent — Reflexion 自修复
    # ==========================================
    max_retries = 3
    attempt = 0
    execution_error = "Initial"
    data_insights = ""
    clean_code = ""
    reflexion_memory = []
    captured_output = io.StringIO()

    while attempt < max_retries and execution_error:
        attempt += 1
        memory_str = "\n".join([f"【失败反思】{m['error']} -> 修复策略: {m['fix_strategy']}" for m in reflexion_memory]) if reflexion_memory else "首次尝试。"

        print(f"\n\n===========================================")
        print(f"[*] 👨‍💻 程序员 Agent 开始写代码 (第 {attempt}/{max_retries} 次)")
        print(f"===========================================\n")

        code_prompt = ChatPromptTemplate.from_messages([
            ("system", """你是顶级数据分析 Python 程序员。
当前已加载数据框 `df`。表头: [{columns_str}]
【本次分析计划】\n{analysis_plan}
【历史修复记忆】\n{memory_str}

【编码绝对规范 - 违反将导致系统崩溃】
1. 必须 import pandas as pd, import matplotlib.pyplot as plt
2. 【终极数据清洗】无论数值列带有千分位逗号(,)、百分号(%)还是货币符号，计算前必须使用正则全部剥离！
   严格示例：df['列名'] = pd.to_numeric(df['列名'].astype(str).str.replace(r'[^\d.-]', '', regex=True), errors='coerce')
3. 必须包含中文字体配置：plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']; plt.rcParams['axes.unicode_minus'] = False
4. 必须画图！每画完一张图，严格使用 `plt.savefig('chart_1.png')` 保存，并立刻调用 `plt.clf()` 清空画布！
5. 【核心要求】算出数值后，必须用 print() 打印出来！没有 print 就等于没有数据！
6. 只输出纯 Python 代码，绝对禁止出现中文全角标点符号！"""),
            ("user", "需求：{query}")
        ])

        try:
            raw_code = ""
            for chunk in (code_prompt | llm_coder).stream({"columns_str": columns_str, "analysis_plan": analysis_plan, "memory_str": memory_str, "query": user_query}):
                raw_code += chunk.content
                print(chunk.content, end="", flush=True)
            
            # 【核心修复点：全角标点终极粉碎机】
            clean_code = raw_code.replace("```python", "").replace("```", "").strip()
            # 暴力把中文的 逗号、冒号、括号 替换为 Python 认可的英文标点
            clean_code = clean_code.replace("，", ",").replace("：", ":").replace("（", "(").replace("）", ")").replace("“", '"').replace("”", '"')
            
        except Exception as e:
            execution_error = f"API异常: {e}"
            continue

        captured_output = io.StringIO()
        execution_error = None
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as _plt
            _plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
            _plt.rcParams["axes.unicode_minus"] = False

            with redirect_stdout(captured_output):
                exec_env = {"df": df, "pd": pd, "matplotlib": matplotlib, "plt": _plt, "os": os}
                exec(clean_code, exec_env)
            print(f"\n[+] 执行成功！图表生成状况: {glob.glob('chart_*.png')}")
        except Exception as e:
            execution_error = str(e)
            print(f"\n[-] 报错：{execution_error}")
            
            # Reflexion
            reflect_prompt = ChatPromptTemplate.from_messages([("system", "用一句话概括报错原因及修复代码。"), ("user", f"报错: {execution_error}\n代码片段: {clean_code[-1000:]}")])
            try:
                reflect_res = (reflect_prompt | llm_coder).invoke({}).content
                reflexion_memory.append({"attempt": attempt, "error": execution_error, "fix_strategy": reflect_res})
            except:
                reflexion_memory.append({"attempt": attempt, "error": execution_error, "fix_strategy": "正则清洗数值列，检查语法"})

    data_insights = captured_output.getvalue()
    generated_charts = glob.glob("chart_*.png")

    if execution_error:
        data_insights = f"【系统致命错误】代码历经 {max_retries} 次抢修依然崩溃！最终报错信息：{execution_error}"
        chart_status = "未生成任何图表，严禁使用 [CHART] 占位符！"
    else:
        chart_status = f"生成的真实图表文件有：{generated_charts}。只能使用对应的占位符！"

    # ==========================================
    # Node 3：🧑‍💼 分析师 Agent — 灵活应变处理
    # ==========================================
    print(f"\n===========================================")
    print("[*] 🧑‍💼 分析师 Agent 正在撰写报告...")
    print(f"===========================================\n")

    analyst_prompt = ChatPromptTemplate.from_messages([
        ("system", """你是资深业务分析师。
【分析计划】\n{analysis_plan}
【运行数据】\n{data_insights}
【系统图表状态】\n{chart_status}

【极其重要的双轨处理策略】
情况 A（代码运行崩溃）：如果【运行数据】中包含“系统致命错误”或报错信息，你**绝对禁止**按原计划生成带有 `[需填充数据]` 的虚假分析！你必须直接输出一份“数据诊断说明”，通俗解释为什么分析失败（比如数据格式错误、逗号未清除等），绝不允许写任何图表占位符！
情况 B（运行成功）：严格按照分析计划的维度解读 `print()` 出来的数据，并在文字说明后插入【系统图表状态】中真实存在的 `[CHART_X]` 占位符。严禁凭空捏造图表名！"""),
        ("user", "原需求：{query}")
    ])

    draft_report = ""
    try:
        for chunk in (analyst_prompt | llm_analyst).stream({"analysis_plan": analysis_plan, "data_insights": data_insights, "chart_status": chart_status, "query": user_query}):
            draft_report += chunk.content
            print(chunk.content, end="", flush=True)
    except Exception as e:
        draft_report = f"草稿生成失败: {e}"

    # ==========================================
    # Node 4：⚖️ 检察官 Agent — 格式清洗
    # ==========================================
    print(f"\n\n===========================================")
    print("[*] ⚖️ 检察官 Agent 正在质检报告...")
    print(f"===========================================\n")

    judge_prompt = ChatPromptTemplate.from_messages([
        ("system", """你是冷酷的报告质检总监。
任务：保留核心报告内容，删掉草稿中的所有 AI 废话。
【强制要求】必须将纯净版 Markdown 包裹在 `<FINAL_REPORT>` 和 `</FINAL_REPORT>` 之间！"""),
        ("user", "【草稿】\n{draft}")
    ])

    try:
        raw_judged = ""
        for chunk in (judge_prompt | llm_judge).stream({"draft": draft_report}):
            raw_judged += chunk.content
            print(chunk.content, end="", flush=True)
        match = re.search(r'<FINAL_REPORT>\s*(.*?)\s*</FINAL_REPORT>', raw_judged, re.DOTALL)
        final_markdown = match.group(1).strip() if match else raw_judged.strip()
    except Exception as e:
        final_markdown = draft_report

    print("\n[*] ⚙️ 渲染引擎：正在打包 HTML...")
    html_string = generate_html_report(final_markdown, report_out)

    return html_string, report_out
