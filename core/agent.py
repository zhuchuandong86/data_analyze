import os
import sys
import io
import re
import glob
import tempfile
from contextlib import redirect_stdout
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from utils.reporter import generate_html_report
import pandas as df



def run_agent_pipeline(df: pd.DataFrame, user_query: str, api_key: str, api_base: str):
    report_out = "final_report.html"

    # ==========================================
    # 【数据预处理】清理表头
    # ==========================================
    df.columns = df.columns.str.strip().str.replace('\n', '').str.replace('\r', '')
    columns_str = ", ".join(df.columns)
    sample_str = df.head(3).to_string()
    
    if not user_query or not user_query.strip():
        user_query = f"对数据全面分析，字段包括：{columns_str}。计算各数值列的统计摘要，画出关键指标对比图表。"
        print(f"[i] 用户未输入需求，使用默认指令。")

    # ==========================================
    # 🌟 异构模型配置
    # ==========================================
    llm_coder = ChatOpenAI(
        temperature=0, model="deepseek-v3-0324",
        api_key=api_key, base_url=api_base, timeout=120
    )
    llm_planner = ChatOpenAI(
        temperature=0, model="deepseek-v3-0324",
        api_key=api_key, base_url=api_base, timeout=60
    )
    llm_analyst = ChatOpenAI(
        temperature=0.3, model="deepseek-v3-0324",
        api_key=api_key, base_url=api_base, timeout=120
    )
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
    # ==========================================
    max_retries = 3
    attempt = 0
    execution_error = "Initial"
    data_insights = ""
    clean_code = ""

    chart_dir = tempfile.mkdtemp(prefix="agent_charts_")
    print(f"[i] 图表临时目录: {chart_dir}")

    reflexion_memory = []
    captured_output = io.StringIO()

    while attempt < max_retries and execution_error:
        attempt += 1

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

        # 【新增/优化】：强化预处理法则，要求引入 re，并防守型编程
        code_prompt = ChatPromptTemplate.from_messages([
            ("system", """你是顶级数据分析 Python 程序员，擅长复杂数据分析。
当前已加载数据框 `df`。表头: [{columns_str}]

【数据预处理黄金法则】（极其重要）
在进行任何计算前，如果数值列（如金额、价格）包含中文或特殊符号（如 '151.11(已退款)'）：
必须使用正则提取核心数字，推荐写法：`df['列名'] = df['列名'].astype(str).str.extract(r'(-?\d+\.?\d*)')[0].astype(float)`。切忌使用简单的 replace。
遇到类型转换，尽量使用 pd.to_numeric(..., errors='coerce') 把无法解析的变成 NaN，避免程序崩溃。

【本次分析计划】
{analysis_plan}

【Reflexion 历史修复记忆】（认真阅读，避免重蹈覆辙）
{memory_str}

【编码规范】
1. 必须 import pandas as pd, import matplotlib.pyplot as plt, import re
2. 中文字体防乱码：
   plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
   plt.rcParams['axes.unicode_minus'] = False
3. 图表保存为 'chart_1.png'（依次递增），绝不使用 plt.show()。
4. 代码中绝对不能出现中文全角标点符号！
5. 【核心要求】每算出一个关键数值，必须立刻 print() 出来！格式示例：
   print(f"下行速率均值: {{avg_val:.2f}}")
   分析师只能看到 print 输出，没有 print 就等于没有数据！
6. 只输出纯 Python 代码，不要任何 markdown 或注释！"""),
            # 【新增/优化】：将真实数据样本喂给 Coder，避免盲目猜测数据格式
            ("user", "分析需求：{query}\n\n【重要】真实数据样本预览：\n{sample}")
        ])

        try:
            raw_code = ""
            for chunk in (code_prompt | llm_coder).stream({
                "columns_str": columns_str,
                "analysis_plan": analysis_plan,
                "memory_str": memory_str,
                "query": user_query,
                "sample": sample_str  # 传入样本
            }):
                raw_code += chunk.content
                print(chunk.content, end="", flush=True)

            print("\n\n[+] 代码接收完毕，开始在沙盒中执行...")
            clean_code = raw_code.replace("```python", "").replace("```", "").strip()
            
            FULLWIDTH_MAP = {
                "，": ",", "。": ".", "：": ":", "；": ";",
                "（": "(", "）": ")", "【": "[", "】": "]",
                "“": '"', "”": '"', "‘": "'", "’": "'",
                "！": "!", "？": "?", "…": "...", "—": "-", "·": ".",
            }
            for zh, en in FULLWIDTH_MAP.items():
                clean_code = clean_code.replace(zh, en)
            
            first_line = clean_code.strip().splitlines()[0] if clean_code.strip() else ""
            if first_line and "一" <= first_line[0] <= "鿿":
                execution_error = f"LLM输出了聊天文字而非代码：{first_line[:50]}"
                print(f"\n[-] {execution_error}")
                reflexion_memory.append({"attempt": attempt, "error": execution_error,
                    "fix_strategy": "必须直接输出Python代码，禁止聊天或提问！"})
                continue
            
            # 【修复点】：增加 hasattr 判断，防止 Reflexion 多次重试时无限递归死循环
            agg_prefix = (
                "import matplotlib\n"
                "import matplotlib.pyplot as plt\n"
                "plt.switch_backend('agg')\n"
                "plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']\n"
                "plt.rcParams['axes.unicode_minus'] = False\n"
                f"__chart_dir__ = r'{chart_dir}'\n"
                "import os as __os__\n"
                "if not hasattr(plt, '_original_savefig'):\n"
                "    plt._original_savefig = plt.savefig\n"
                "    def __patched_savefig__(fname, *a, **kw):\n"
                "        if not __os__.path.isabs(str(fname)):\n"
                "            fname = __os__.path.join(__chart_dir__, __os__.path.basename(str(fname)))\n"
                "        plt._original_savefig(fname, *a, **kw)\n"
                "    plt.savefig = __patched_savefig__\n"
            )

            clean_code = agg_prefix + clean_code
        except Exception as e:
            execution_error = f"API超时或断开: {e}"
            print(f"\n[-] {execution_error}")
            continue

        # 沙盒执行
        captured_output = io.StringIO()
        execution_error = None
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as _plt
            _plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
            _plt.rcParams["axes.unicode_minus"] = False

            with redirect_stdout(captured_output):
                exec_env = {"df": df, "pd": pd, "os": os, "re": re}
                exec(clean_code, exec_env)
                
            charts_found = glob.glob(os.path.join(chart_dir, "chart_*.png"))
            captured_preview = captured_output.getvalue().strip()
            if captured_preview:
                print(f"[+] 代码执行成功！已生成图表: {charts_found}")
                print(f"   [i] 数据输出预览:\n{captured_preview[:500]}")
            else:
                print(f"[+] 代码执行成功！已生成图表: {charts_found}")
                print("   [!] 警告：代码无任何 print 输出，分析师将无真实数据可用！")
        except Exception as e:
            execution_error = str(e)
            print(f"[-] 执行报错：{execution_error}")

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
                    "code": clean_code[-2000:]
                }):
                    reflect_raw += chunk.content
                    print(chunk.content, end="", flush=True)

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
    # 【新增/优化】：梳理产物 & 熔断机制 (Fail-Fast)
    # 彻底拦截失败任务，不让后续模型浪费 Token 写无用报告
    # ==========================================
    data_insights = captured_output.getvalue()
    generated_charts = glob.glob(os.path.join(chart_dir, "chart_*.png"))

    if execution_error:
        print(f"\n[!] 触发熔断机制：代码修复超过 {max_retries} 次依然失败，立即停止后续 Agent 调用以节省 Token。")
        
        # 构建一个优雅的降级 HTML 错误报告
        memory_html = "".join([
            f"<li style='margin-bottom:8px;'><b>第{m['attempt']}次反思：</b> {m['root_cause']}<br><i style='color:#666;'>对策：{m['fix_strategy']}</i></li>" 
            for m in reflexion_memory
        ])
        
        error_html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head><meta charset="UTF-8"><title>数据分析中断</title></head>
<body style="font-family: 'Segoe UI', sans-serif; padding: 20px; background-color: #f4f7f6;">
    <div style="max-width: 900px; margin: 0 auto; background: #fff; padding: 30px; border-radius: 8px; border-top: 5px solid #e74c3c; box-shadow: 0 4px 15px rgba(0,0,0,0.05);">
        <h2 style="color: #c0392b; margin-top: 0;">⚠️ 数据分析中断 (熔断保护生效)</h2>
        <p>AI 程序员经过 <b>{max_retries}</b> 次自我反思与代码重写，仍未能完全解决数据清洗/计算过程中的报错。为节省计算资源 (Token)，已自动停止后续分析报告的生成。</p>
        
        <h4 style="color: #2c3e50; border-bottom: 1px solid #eee; padding-bottom: 5px;">最后一次致命报错：</h4>
        <pre style="background: #fdf2f2; padding: 15px; border-radius: 6px; overflow-x: auto; color: #c0392b; font-size: 14px;">{execution_error}</pre>
        
        <h4 style="color: #2c3e50; border-bottom: 1px solid #eee; padding-bottom: 5px; margin-top: 25px;">🧠 AI 努力过的反思记录：</h4>
        <ul style="background: #f8f9fa; padding: 15px 35px; border-radius: 6px; font-size: 14px; line-height: 1.6;">
            {memory_html}
        </ul>
        
        <div style="margin-top: 30px; padding: 15px; background: #e8f4f8; border-radius: 6px; color: #2980b9;">
            <h4 style="margin: 0 0 10px 0;">💡 人工干预建议：</h4>
            <p style="margin: 0; font-size: 14px;">您的数据中可能存在过于复杂的业务混合格式（如金额列中带有大量文本说明）。建议在左侧“分析需求”框中明确提示 AI，例如：<b>“清洗金额字段时，请用正则表达式强制提取括号前的数字作为浮点数，忽略无法转换的值。”</b></p>
        </div>
    </div>
</body>
</html>"""
        
        # 将错误报告写入文件并直接 return（熔断）
        with open(report_out, "w", encoding="utf-8") as f:
            f.write(error_html)
        return error_html, report_out

    # 如果代码执行成功，才记录生成的图表
    chart_status = f"生成的图表文件有：{[os.path.basename(c) for c in generated_charts]}。只能使用对应的占位符！"

    # ==========================================
    # Node 3：🧑‍💼 分析师 Agent — 参考 Planner 计划写报告
    # ==========================================
    print(f"\n===========================================")
    print("[*] 🧑‍💼 分析师 Agent 正在按计划撰写报告...")
    print(f"===========================================\n")

    analyst_prompt = ChatPromptTemplate.from_messages([
        ("system", """你是资深业务分析师。

【铁律：禁止数据幻觉】——违反此条视为系统故障
你只能使用下方【运行数据】中 print() 真实输出的数字。
禁止自行编造任何数值、百分比、金额、笔数、排名！
如果某个数字不在【运行数据】中，必须写"（数据未输出）"，绝不猜测！

【分析计划】
{analysis_plan}

【运行数据】（这是代码实际 print 的内容，是唯一真实数据来源）
{data_insights}

【图表状态】
{chart_status}

处理规则：
- 若【运行数据】有真实数字：按分析计划解读，只插入【图表状态】中真实存在的 [CHART_X]
- 若【运行数据】为空：直接写"代码未输出任何数据，无法生成报告。"，停止输出"""),
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
    # Node 4：⚖️ 检察官 Agent — 格式清洗
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
    import shutil
    copied = []
    for src in glob.glob(os.path.join(chart_dir, "chart_*.png")):
        dst = os.path.basename(src)
        shutil.copy2(src, dst)
        copied.append(dst)
    if copied:
        print(f"[i] 已将图表复制到工作目录: {copied}")

    print("\n[*] ⚙️ 渲染引擎：正在打包 HTML...")
    html_string = generate_html_report(final_markdown, report_out)

    return html_string, report_out
