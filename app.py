import os
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from core.agent import run_agent_pipeline, run_followup_chat  # 【修改】引入新增的追问函数


# 加载环境变量
load_dotenv()
API_KEY = os.getenv("INTERNAL_API_KEY")
API_BASE = os.getenv("INTERNAL_API_BASE")

# 页面配置
st.set_page_config(page_title="AI 数据分析终端", page_icon="📈", layout="wide")

st.title("📈 智能数据分析与洞察终端")
st.markdown("支持电信网络、财务及通用数据的自动化扫描与可视化。")

# 【新增】初始化 Session State 用于保存页面状态，防止聊天时报告消失
if 'report_html' not in st.session_state:
    st.session_state.report_html = None
if 'report_path' not in st.session_state:
    st.session_state.report_path = None
if 'context_data' not in st.session_state:
    st.session_state.context_data = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# 侧边栏：文件上传
with st.sidebar:
    st.header("1. 上传数据")
    uploaded_file = st.file_uploader("支持 CSV 或 Excel 文件", type=['csv', 'xlsx', 'xls'])
    
    st.header("2. 分析需求 (可选)")
    user_query = st.text_area("请输入您的具体关注点...", placeholder="例如：分析各分公司的利润趋势，或按默认策略全面扫描。")
    
    analyze_btn = st.button("🚀 开始智能分析", type="primary")

# 主界面区域：执行分析逻辑
if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        st.write("### 📄 数据预览", df.head(3))
    except Exception as e:
        st.error(f"读取文件失败: {e}")
        st.stop()

    if analyze_btn:
        if not API_KEY:
            st.error("缺失 API KEY 配置，请检查 .env 文件！")
            st.stop()
            
        with st.spinner('🤖 AI 正在拼命敲代码、画图并思考中，请稍候...'):
            # 【修改】接收第三个返回值 context_data
            html_content, report_path, context_data = run_agent_pipeline(df, user_query, API_KEY, API_BASE)
            
            # 【新增】将生成的结果存入 session_state
            st.session_state.report_html = html_content
            st.session_state.report_path = report_path
            st.session_state.context_data = context_data
            st.session_state.chat_history = []  # 每次重新分析时，清空之前的聊天记录
else:
    st.info("👈 请先在左侧上传数据文件。")


# ==============================================================
# 【新增】：如果报告已经生成，渲染报告并在下方开启“追问与优化”模块
# ==============================================================
if st.session_state.report_html:
    st.success("✅ 分析完成！")
    
    # 渲染 HTML 报告
    st.components.v1.html(st.session_state.report_html, height=800, scrolling=True)
    
    with open(st.session_state.report_path, "r", encoding="utf-8") as f:
        st.download_button(
            label="📥 下载独立 HTML 报告",
            data=f.read(),
            file_name="AI_Analysis_Report.html",
            mime="text/html"
        )
        
    st.divider()
    st.markdown("### 💬 报告深度追问与优化")
    st.caption("您可以基于上方报告继续提问，例如：'帮我深挖一下第二部分的数据'，或 '将报告的结论部分改写得更委婉一些'。")
    
    # 渲染历史聊天记录
    for msg in st.session_state.chat_history:
        st.chat_message(msg["role"]).write(msg["content"])
        
    # 处理用户追问输入
    if prompt := st.chat_input("在此输入您的追问或优化需求..."):
        # 记录用户问题并显示
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        
        # 调用追问 Agent 生成回复
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""
            try:
                # 排除当前刚输入的一条作为历史
                history = st.session_state.chat_history[:-1]
                stream_generator = run_followup_chat(
                    user_query=prompt, 
                    chat_history=history, 
                    context_data=st.session_state.context_data, 
                    api_key=API_KEY, 
                    api_base=API_BASE
                )
                
                # 实现流式打字机效果
                # 实现流式打字机效果 (增加节流机制，防止 WebSocket 崩溃)
                count = 0
                for chunk in stream_generator:
                    full_response += chunk.content
                    count += 1
                    # 每积攒 8 个 token 才向前端发送一次刷新请求，极大减轻 WebSocket 压力
                    if count % 8 == 0:
                        response_placeholder.markdown(full_response + "▌")
                # 循环结束后输出完整内容并去掉光标
                response_placeholder.markdown(full_response)
                
            except Exception as e:
                full_response = f"追问请求失败: {e}"
                st.error(full_response)
                
        # 保存 AI 回复
        st.session_state.chat_history.append({"role": "assistant", "content": full_response})
