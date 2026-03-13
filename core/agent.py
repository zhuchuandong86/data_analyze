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
    sample_str = df.head(3).to_string()  # 给 Planner 看几行样本，帮助判断数据类型

    # ==========================================
    # 🌟 异构模型配置
    # ==========================================
    # 程序员：专职写代码，Temperature=0 保证严谨
    llm_coder = ChatOpenAI(
        temperature=0, model="qwen2.5-coder-32b-instruct",
        api_key=api_key, base_url=api_base, timeout=120
    )
    # 规划师：轻量快速，只需要输出结构化计划
    llm_planner = ChatOpenAI(
        temperature=0, model="deepseek-v3-0324",
        api_key=api_key, base_url=api_base, timeout=60
    )
    # 分析师：写报告，稍微发散
    llm_analyst = ChatOpenAI(
        temperature=0.3, model="deepseek-v3-0324",
        api_key=api_key, base_url=api_base, timeout=120
    )
    # 检察官：格式清洗，换成轻量模型节省成本（原来用 r1 太贵了）
    llm_judge = ChatOpenAI(
        temperature=0, model="deepseek-v3-0324",
        api_key=api_key, base_url=api_base, timeout=120
    )

    # ==========================================
    # Node 0：🗺️ Planner Agent — 动态制定分析计划
    # ==========================================
    print("\n===========================================")
    print("[*] 🗺️ Planner Agent 正在分析数据，制定分析计划...")
    print("===========================================\n")

    planner_prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一个数据分析规划师。根据用户提供的数据表信息，制定一份简洁的分析计划。

【输出格式要求】严格按如下格式输出，不要多余废话：
数据类型: <电信网络数据 / 销售业务数据 / 财务数据 / 通用数据>
分析维度:
1. <维度一，10字以内>
2. <维度二，10字以内>
3. <维度三，10字以内>
核心指标: <列出3-5个最关键的字段名>
图表建议: <建议生成哪几类图表，一行内说完>
质差/异常排查重点: <如果是电信数据，写出重点排查方向；否则写"通用异常检测">"""),
        ("user", "表头: {columns}\n\n数据样本:\n{sample}\n\n用户需求: {query}")
    ])

    analysis_plan = ""
    try:
        for chunk in (planner_prompt | llm_planner).stream({
            "columns": columns_str,
            "sample": sample_str,
            "query": user_query
        }):
            analysis_plan += chunk.content
            print(chunk.content, end="", flush=True)
        print("\n\n[+] 分析计划制定完毕！")
    except Exception as e:
        analysis_plan = "通用数据分析：趋势、对比、异常检测"
        print(f"\n[-] Planner 调用失败，使用默认计划：{e}")

    # ==========================================
    # Node 1 & 2：👨‍💻 程序员 Agent — Reflexion 自修复
    # 改造点：每次失败后先"反思"，生成结构化修复记忆，
    #         带着经验重写，而不是盲目粘贴报错重试。
    # ==========================================
    max_retries = 3
    attempt = 0
    execution_error = "Initial"
    data_insights = ""
    clean_code = ""

    # Reflexion 记忆列表：每次失败都往里追加一条结构化反思
    reflexion_memory = []
    # 必须在循环外初始化！若每次都在 API 阶段 continue，循环内的赋值永远不会执行
    captured_output = io.StringIO()

    while attempt < max_retries and execution_error:
        attempt += 1

        # 把历史反思记忆格式化给 Coder 看
        if reflexion_memory:
            memory_str = "\n".join([
                f"【第{m['attempt']}次失败反思】\n"
                f"  错误信息: {m['error']}\n"
                f"  根因判断: {m['root_cause']}\n"
                f"  修复策略: {m['fix_strategy']}\n"
                f"  禁止重蹈: {m['avoid']}"
                for m in reflexion_memory
            ])
        else:
            memory_str = "无历史报错，首次尝试。"

        print(f"\n===========================================")
        print(f"[*] 👨‍💻 程序员 Agent 开始写代码 (第 {attempt}/{max_retries} 次，携带 {len(reflexion_memory)} 条反思记忆)")
        print(f"===========================================\n")

        code_prompt = ChatPromptTemplate.from_messages([
            ("system", """你是顶级数据分析 Python 程序员，擅长无线网络数据分析。
当前已加载数据框 `df`。表头: [{columns_str}]

【本次分析计划】（Planner 已制定，请严格遵照执行）
{analysis_plan}

【Reflexion 历史修复记忆】（认真阅读，避免重蹈覆辙）
{memory_str}

【编码规范】
1. 必须 import pandas as pd, import matplotlib.pyplot as plt
2. 中文字体防乱码：
   plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
   plt.rcParams['axes.unicode_minus'] = False
3. 含货币符或字符的数值列，先用正则清洗转为数字。
4. 图表保存为 'chart_1.png'（依次递增），绝不使用 plt.show()。
5. 代码中绝对不能出现中文全角标点符号！
6. 【核心要求】每算出一个关键数值，必须立刻 print() 出来！格式示例：
   print(f"4G忙时平均下行速率: {avg_speed:.2f} Mbps")
   print(f"PRB利用率超80%小区数: {high_prb_count} 个，占比 {ratio:.1%}")
   分析师只能看到 print 输出，没有 print 就等于没有数据，报告会全是 XX！
7. 只输出纯 Python 代码，不要任何 markdown 或注释！"""),
            ("user", "分析需求：{query}")
        ])

        try:
            raw_code = ""
            for chunk in (code_prompt | llm_coder).stream({
                "columns_str": columns_str,
                "analysis_plan": analysis_plan,
                "memory_str": memory_str,
                "query": user_query
            }):
                raw_code += chunk.content
                print(chunk.content, end="", flush=True)

            print("\n\n[+] 代码接收完毕，开始在沙盒中执行...")
            clean_code = raw_code.replace("```python", "").replace("```", "").strip()
        except Exception as e:
            execution_error = f"API超时或断开: {e}"
            print(f"\n[-] {execution_error}")
            continue

        # 沙盒执行
        captured_output = io.StringIO()
        execution_error = None
        try:
            # 预注入 Agg 后端的 matplotlib/plt
            # Windows + Streamlit headless 环境下必须强制 Agg，否则 savefig 静默失败
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as _plt
            _plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
            _plt.rcParams["axes.unicode_minus"] = False

            chart_dir = os.path.abspath(".")
            print(f"   [i] 图表将保存至: {chart_dir}")

            with redirect_stdout(captured_output):
                exec_env = {
                    "df": df, "pd": pd,
                    "matplotlib": matplotlib, "plt": _plt,
                    "os": os,
                }
                exec(clean_code, exec_env)
            charts_found = glob.glob("chart_*.png")
            print(f"[+] 代码执行成功！已生成图表: {charts_found if charts_found else "无"}")
        except Exception as e:
            execution_error = str(e)
            print(f"[-] 执行报错：{execution_error}")

            # ======================================
            # 🔁 Reflexion：失败后先反思，再重写
            # 让 Coder 自己分析根因并制定修复策略
            # ======================================
            print("\n[*] 🔁 Reflexion：正在反思失败原因，生成修复记忆...")

            reflect_prompt = ChatPromptTemplate.from_messages([
                ("system", """你是代码调试专家。分析下面的报错，输出结构化反思。
严格按如下格式，每项一行，不要多余内容：
根因判断: <一句话说清楚为什么报错>
修复策略: <具体应该怎么改，越具体越好>
禁止重蹈: <下次写代码时必须避免的做法>"""),
                ("user", "报错信息: {error}\n\n出错代码片段:\n{code}")
            ])

            reflection = {"attempt": attempt, "error": execution_error,
                          "root_cause": "", "fix_strategy": "", "avoid": ""}
            try:
                reflect_raw = ""
                for chunk in (reflect_prompt | llm_coder).stream({
                    "error": execution_error,
                    "code": clean_code[-2000:]  # 只取末尾 2000 字符避免 token 超限
                }):
                    reflect_raw += chunk.content
                    print(chunk.content, end="", flush=True)

                # 解析反思结果
                for line in reflect_raw.splitlines():
                    if line.startswith("根因判断:"):
                        reflection["root_cause"] = line.replace("根因判断:", "").strip()
                    elif line.startswith("修复策略:"):
                        reflection["fix_strategy"] = line.replace("修复策略:", "").strip()
                    elif line.startswith("禁止重蹈:"):
                        reflection["avoid"] = line.replace("禁止重蹈:", "").strip()

                print(f"\n[+] 反思完成，记忆已写入（第 {attempt} 条）")
            except Exception as re_err:
                reflection["root_cause"] = execution_error
                reflection["fix_strategy"] = "检查语法和数据类型"
                reflection["avoid"] = "避免假设列名和数据格式"
                print(f"\n[-] 反思模型调用失败，使用默认记忆：{re_err}")

            reflexion_memory.append(reflection)

    # ==========================================
    # 梳理产物
    # ==========================================
    data_insights = captured_output.getvalue()
    generated_charts = glob.glob("chart_*.png")

    if execution_error:
        data_insights = (
            f"【系统提示】代码经过 {max_retries} 次 Reflexion 修复依然失败。\n"
            f"最终报错：{execution_error}\n"
            f"历史反思记录：\n" +
            "\n".join([f"  第{m['attempt']}次 - {m['root_cause']}" for m in reflexion_memory])
        )
        chart_status = "未生成任何图表，严禁使用 [CHART] 占位符！"
    else:
        chart_status = f"生成的图表文件有：{generated_charts}。只能使用对应的占位符！"

    # ==========================================
    # Node 3：🧑‍💼 分析师 Agent — 参考 Planner 计划写报告
    # ==========================================
    print(f"\n===========================================")
    print("[*] 🧑‍💼 分析师 Agent 正在按计划撰写报告...")
    print(f"===========================================\n")

    analyst_prompt = ChatPromptTemplate.from_messages([
        ("system", """你是资深业务分析师。
【分析计划】\n{analysis_plan}
【运行数据】\n{data_insights}
【图表状态】\n{chart_status}
任务：严格按照分析计划的维度组织报告结构，深度解读数据，在段落中合理插入图表占位符（如 `[CHART_1]`）。"""),
        ("user", "原需求：{query}")
    ])

    draft_report = ""
    try:
        for chunk in (analyst_prompt | llm_analyst).stream({
            "analysis_plan": analysis_plan,
            "data_insights": data_insights,
            "chart_status": chart_status,
            "query": user_query
        }):
            draft_report += chunk.content
            print(chunk.content, end="", flush=True)
        print("\n\n[+] 草稿撰写完毕！")
    except Exception as e:
        draft_report = f"草稿生成失败: {e}"

    # ==========================================
    # Node 4：⚖️ 检察官 Agent — 格式清洗（换轻量模型）
    # ==========================================
    print(f"\n===========================================")
    print("[*] ⚖️ 检察官 Agent 正在质检报告...")
    print(f"===========================================\n")

    judge_prompt = ChatPromptTemplate.from_messages([
        ("system", """你是冷酷的报告质检总监。
任务：对草稿进行"脱水"。删掉所有 AI 废话、内心戏和免责声明。保留所有数据、结论和图表占位符。
【强制要求】必须将最终纯净版 Markdown 报告包裹在 `<FINAL_REPORT>` 和 `</FINAL_REPORT>` 标签之间！"""),
        ("user", "【草稿】\n{draft}")
    ])

    final_markdown = draft_report
    try:
        raw_judged = ""
        for chunk in (judge_prompt | llm_judge).stream({"draft": draft_report}):
            raw_judged += chunk.content
            print(chunk.content, end="", flush=True)

        print("\n\n[+] 质检完毕，提取精华...")
        match = re.search(r'<FINAL_REPORT>\s*(.*?)\s*</FINAL_REPORT>', raw_judged, re.DOTALL)
        final_markdown = match.group(1).strip() if match else raw_judged.strip()
    except Exception as e:
        print(f"[-] 检察官调用失败，使用草稿原文：{e}")

    # ==========================================
    # Node 5：⚙️ 渲染引擎
    # ==========================================
    print("\n[*] ⚙️ 渲染引擎：正在打包 HTML...")
    html_string = generate_html_report(final_markdown, report_out)

    return html_string, report_out
