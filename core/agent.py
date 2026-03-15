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
import streamlit as st
from datetime import datetime


def run_agent_pipeline(df: pd.DataFrame, user_query: str, api_key: str, api_base: str):
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_out = f"final_report_{current_time}.html"

    # ==========================================
    # 【数据预处理】终极数据洗手池 
    # ==========================================
    df.dropna(how='all', inplace=True)
    df.dropna(axis=1, how='all', inplace=True)
    df = df.loc[:, ~df.columns.astype(str).str.contains('^Unnamed')]
    df.columns = df.columns.astype(str).str.strip().str.replace('\n', '').str.replace('\r', '').str.replace('　', '')
    
    if df.empty or len(df.columns) == 0:
        error_html = "<h2 style='color:red;'>❌ 数据读取失败</h2><p>表格无有效数据。</p>"
        with open(report_out, "w", encoding="utf-8") as f: f.write(error_html)
        return error_html, report_out, {}  # 【修改】增加第三个返回值，返回空上下文

    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].replace(['-', '--', '无', 'N/A', 'NA', 'null', ''], pd.NA)
            temp_col = df[col].astype(str).str.replace(r'[¥$,\s]', '', regex=True)
            converted = pd.to_numeric(temp_col, errors='coerce')
            if converted.notna().mean() > 0.5:
                df[col] = converted

    columns_str = ", ".join(df.columns)
    sample_str = df.head(5).fillna("空值(NaN)").to_string()
    
    if not user_query or not user_query.strip():
        user_query = f"对数据全面分析，字段包括：{columns_str}。计算各数值列的统计摘要，画出关键指标对比图表。"
        st.info("💡 用户未输入需求，系统使用默认探索指令。")

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

    # ==========================================
    # Node 0：🗺️ Planner Agent
    # ==========================================
    st.markdown("### 🗺️ Planner Agent 正在分析数据，制定分析计划...")

    planner_prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一名顶尖的数据科学家。请对输入数据进行专业的 EDA (探索性数据分析) 和 KDD (知识发现) 规划。

【输出格式要求】严格按如下格式输出：
数据类型: <业务场景识别>
分析思路：模式 → 假设 → 驱动因素 → 洞察 → 行动建议
深度分析维度:
1. 描述性分析: <指出需要统计哪些核心宏观指标和分布特征>
2. 诊断性分析: <指出可能存在的异常点，并提出根因分析(RCA)的方向>
3. 预测与指导性分析: <提出业务优化建议、资源分配方向或聚类分层建议>
核心指标: <列出3-5个最关键的字段名>
图表建议: <列出专业图表，如"双变量相关性散点图", "异常值箱线图", "核心指标趋势折线图">
数据挖掘挖掘点: <如果进行特征工程或关联分析，最值得挖掘的变量是什么？>"""),
        ("user", "表头: {columns}\n\n数据样本:\n{sample}\n\n用户需求: {query}")
    ])

    analysis_plan = ""
    try:
        plan_placeholder = st.empty()
        count = 0
        for chunk in (planner_prompt | llm_planner).stream({
            "columns": columns_str, "sample": sample_str, "query": user_query
        }):
            analysis_plan += chunk.content
            count += 1
            if count % 8 == 0:  # 节流
                plan_placeholder.markdown(f"```text\n{analysis_plan}▌\n```")
        plan_placeholder.markdown(f"```text\n{analysis_plan}\n```")
        st.success("✅ 分析计划制定完毕！")
    except Exception as e:
        analysis_plan = "通用数据分析：趋势、对比、异常检测"
        st.error(f"❌ Planner 调用失败，使用默认计划：{e}")

    # ==========================================
    # Node 1 & 2：👨‍💻 程序员 Agent
    # ==========================================
    max_retries = 3
    attempt = 0
    execution_error = "Initial"
    data_insights = ""
    clean_code = ""
    chart_dir = tempfile.mkdtemp(prefix="agent_charts_")
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

        st.markdown(f"### 👨‍💻 程序员 Agent 开始写代码 (第 {attempt}/{max_retries} 次)")
        if reflexion_memory:
            st.caption(f"已携带 {len(reflexion_memory)} 条反思记忆")

        # 【修改】：防崩溃铁律升级，强调“独立容错，继续执行”
        code_prompt = ChatPromptTemplate.from_messages([
            ("system", """你是顶级数据分析 Python 程序员。
当前已加载数据框 `df`。表头: [{columns_str}]

【本次分析计划】
{analysis_plan}

【历史修复记忆】
{memory_str}

【编码规范】(必须严格遵守)
1. 必须 import pandas as pd, import matplotlib.pyplot as plt, import re
2. 中文字体防乱码：
   plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
   plt.rcParams['axes.unicode_minus'] = False
3. 图表保存为 'chart_1.png'（依次递增），绝不使用 plt.show()。
4. 【防崩溃铁律-极限容错】：因为数据可能极度不规范，**你必须为每一个独立的指标计算、每一张独立图表的生成，分别使用独立的 try-except 块包裹！**
   即使某个图表或指标报错，在 except 中 print 错误信息后，程序**必须继续往下执行**去画下一张图！绝不能让整个脚本中断！
   示例：
   try:
       # 计算和画图逻辑 A
   except Exception as e:
       print(f"部分生成失败: {{e}}")
   try:
       # 计算和画图逻辑 B
   except Exception as e:
       pass
5. 每算出一个关键数值，必须 print()！分析师只能看到 print 输出！
6. 只输出纯 Python 代码，不要任何 markdown 或注释！"""),
            ("user", "分析需求：{query}\n\n【重要】真实数据样本预览：\n{sample}")
        ])

        try:
            raw_code = ""
            code_placeholder = st.empty()
            count = 0
            for chunk in (code_prompt | llm_coder).stream({
                "columns_str": columns_str, "analysis_plan": analysis_plan,
                "memory_str": memory_str, "query": user_query, "sample": sample_str
            }):
                raw_code += chunk.content
                count += 1
                if count % 8 == 0:  # 节流
                    code_placeholder.markdown(f"```python\n{raw_code}▌\n```")
            code_placeholder.markdown(f"```python\n{raw_code}\n```")

            st.info("⚙️ 代码接收完毕，开始在沙盒中执行...")
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
                st.error(f"❌ {execution_error}")
                reflexion_memory.append({"attempt": attempt, "error": execution_error,
                    "fix_strategy": "必须直接输出Python代码，禁止聊天或提问！"})
                continue
            
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
            st.error(f"❌ {execution_error}")
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
                exec_env = {"df": df, "pd": pd, "os": os, "re": re}
                exec(clean_code, exec_env)
                
            charts_found = glob.glob(os.path.join(chart_dir, "chart_*.png"))
            captured_preview = captured_output.getvalue().strip()
            if captured_preview:
                st.success(f"✅ 代码执行成功！已生成图表: {[os.path.basename(c) for c in charts_found]}")
                with st.expander("👀 查看运行数据输出 (供分析师参考)"):
                    st.code(captured_preview[:2000] + ("\n...(已省略)" if len(captured_preview)>2000 else ""))
            else:
                st.warning("⚠️ 警告：代码无任何 print 输出，分析师将无真实数据可用！")
        except Exception as e:
            execution_error = str(e)
            st.error(f"❌ 执行报错：{execution_error}")

            st.warning("🔁 Reflexion：正在反思失败原因，生成修复记忆...")
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
                reflect_placeholder = st.empty()
                count = 0
                for chunk in (reflect_prompt | llm_coder).stream({
                    "error": execution_error, "code": clean_code[-2000:]
                }):
                    reflect_raw += chunk.content
                    count += 1
                    if count % 8 == 0:  # 节流
                        reflect_placeholder.markdown(f"```text\n{reflect_raw}▌\n```")
                reflect_placeholder.markdown(f"```text\n{reflect_raw}\n```")

                for line in reflect_raw.splitlines():
                    if line.startswith("根因判断:"):
                        reflection["root_cause"] = line.replace("根因判断:", "").strip()
                    elif line.startswith("修复策略:"):
                        reflection["fix_strategy"] = line.replace("修复策略:", "").strip()
                    elif line.startswith("禁止重蹈:"):
                        reflection["avoid"] = line.replace("禁止重蹈:", "").strip()

                st.success(f"✅ 反思完成，记忆已写入（第 {attempt} 条）")
            except Exception as re_err:
                reflection["root_cause"] = execution_error
                reflection["fix_strategy"] = "检查语法和数据类型"
                reflection["avoid"] = "避免假设列名和数据格式"
                st.error(f"❌ 反思模型调用失败，使用默认记忆：{re_err}")

            reflexion_memory.append(reflection)

    data_insights = captured_output.getvalue()
    generated_charts = glob.glob(os.path.join(chart_dir, "chart_*.png"))

    if execution_error:
        st.error(f"🚨 触发熔断机制：代码修复超过 {max_retries} 次依然失败，立即停止后续 Agent 调用以节省 Token。")
        memory_html = "".join([f"<li style='margin-bottom:8px;'><b>第{m['attempt']}次反思：</b> {m['root_cause']}<br><i style='color:#666;'>对策：{m['fix_strategy']}</i></li>" for m in reflexion_memory])
        error_html = f"<h2>⚠️ 数据分析中断</h2><pre>{execution_error}</pre><ul>{memory_html}</ul>"
        with open(report_out, "w", encoding="utf-8") as f:
            f.write(error_html)
        return error_html, report_out, {} # 【修改】增加第三个返回值

    chart_status = f"生成的图表文件有：{[os.path.basename(c) for c in generated_charts]}。只能使用对应的占位符！"

    # ==========================================
    # Node 3：🧑‍💼 终极分析师 Agent
    # ==========================================
    st.markdown("### 🧑‍💼 分析师 Agent 正在撰写最终洞察报告 (直出模式)...")

    # 【修改】：增加了“部分容错铁律”，不要因为缺图就放弃
    analyst_prompt = ChatPromptTemplate.from_messages([
        ("system", """你是顶级的商业数据咨询顾问。你的任务是基于运行数据，直接输出一篇排版精美的 Markdown 最终商业洞察报告。

【排版与质检铁律】
1. 没有任何废话、内心戏，直接输出报告正文。
2. 必须将整篇报告包裹在 `<FINAL_REPORT>` 和 `</FINAL_REPORT>` 标签之间！
3. 采用“核心结论 -> 数据证据 -> 商业建议”的逻辑框架。

【部分容错铁律】（重要！）
如果【运行数据】中包含某些报错信息（如某图表或指标失败），请自动忽略失败的部分，**全力基于成功生成的有效数据**撰写报告。绝不要因为个别数据的缺失而在报告中抱怨或中断生成！

【图表与数据反幻觉铁律】
1. 只使用【运行数据】中的真实数字，严禁编造！
2. 如果【图表状态】显示有生成的图表（如 chart_1.png），你**必须**在分析逻辑最关键的证据处单独起一行插入 `[CHART_1]`！严禁虚构未生成的图表！

【输入信息】
分析计划: {analysis_plan}
运行数据: {data_insights}
图表状态: {chart_status}"""),
        ("user", "原需求：{query}")
    ])

    final_markdown = ""
    try:
        raw_report = ""
        report_placeholder = st.empty()
        count = 0
        for chunk in (analyst_prompt | llm_analyst).stream({
            "analysis_plan": analysis_plan, "data_insights": data_insights,
            "chart_status": chart_status, "query": user_query
        }):
            raw_report += chunk.content
            count += 1
            if count % 8 == 0:  # 节流
                report_placeholder.markdown(raw_report + "▌")
        report_placeholder.markdown(raw_report)
            
        st.success("✅ 最终报告撰写完毕！")
        match = re.search(r'<FINAL_REPORT>\s*(.*?)\s*</FINAL_REPORT>', raw_report, re.DOTALL)
        final_markdown = match.group(1).strip() if match else raw_report.strip()
    except Exception as e:
        final_markdown = f"报告生成失败: {e}"
        st.error(final_markdown)

    # ==========================================
    # Node 5：⚙️ 渲染引擎
    # ==========================================
    import shutil
    copied = []
    for src in glob.glob(os.path.join(chart_dir, "chart_*.png")):
        dst = os.path.basename(src)
        shutil.copy2(src, dst)
        copied.append(dst)

    html_string = generate_html_report(final_markdown, report_out)

    # 【修改】：将分析上下文打包返回，留给追问 Agent 使用
    context_dict = {
        "plan": analysis_plan,
        "data": data_insights
    }
    return html_string, report_out, context_dict


# 【新增】：追问与优化专属 Agent
def run_followup_chat(user_query: str, chat_history: list, context_data: dict, api_key: str, api_base: str):
    llm_chat = ChatOpenAI(temperature=0.4, model="deepseek-v3-0324", api_key=api_key, base_url=api_base)
    
    system_prompt = """你是一个顶级的数据分析咨询顾问。用户正在就刚刚生成的数据洞察报告向你追问或要求优化。
    
    【后台运行真实数据上下文】
    这是系统刚才跑出来的真实数据统计结果：
    {data}
    
    【回答要求】
    1. 基于上述真实数据回答用户的追问。如果超出数据范围，请明确告知“当前运行数据中未包含该统计”。
    2. 如果用户要求优化报告内容，请直接给出优化后的文案。
    3. 语气保持客观、专业。
    4. 【重要排版规范】：如果你需要输出表格，请务必使用标准的 Markdown 表格格式！表格前后必须各留一个空行，并且必须包含表头分隔线，例如：
    
    | 维度一 | 维度二 |
    |---|---|
    | 数据A | 数据B |
    
    """
    
    messages = [("system", system_prompt.format(data=context_data.get("data", "无数据")))]
    
    # 填入历史对话记录
    for msg in chat_history:
        messages.append((msg["role"], msg["content"]))
        
    messages.append(("user", user_query))
    prompt = ChatPromptTemplate.from_messages(messages)
    
    return (prompt | llm_chat).stream({})
