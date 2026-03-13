import os
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from core.agent import run_agent_pipeline



# 加载环境变量
load_dotenv()
API_KEY = os.getenv("INTERNAL_API_KEY")
API_BASE = os.getenv("INTERNAL_API_BASE")

# 页面配置
st.set_page_config(page_title="AI 数据分析终端", page_icon="📈", layout="wide")

st.title("📈 智能数据分析与洞察终端")
st.markdown("支持电信网络、财务及通用数据的自动化扫描与可视化。")

# 侧边栏：文件上传
with st.sidebar:
    st.header("1. 上传数据")
    uploaded_file = st.file_uploader("支持 CSV 或 Excel 文件", type=['csv', 'xlsx', 'xls'])
    
    st.header("2. 分析需求 (可选)")
    user_query = st.text_area("请输入您的具体关注点...", placeholder="例如：分析各分公司的利润趋势，或按默认策略全面扫描。")
    
    analyze_btn = st.button("🚀 开始智能分析", type="primary")

# 主界面区域
if uploaded_file is not None:
    # 预览数据
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        st.write("### 📄 数据预览", df.head(3))
    except Exception as e:
        st.error(f"读取文件失败: {e}")
        st.stop()

    # 执行分析
    if analyze_btn:
        if not API_KEY:
            st.error("缺失 API KEY 配置，请检查 .env 文件！")
            st.stop()
            
        with st.spinner('🤖 AI 正在拼命敲代码、画图并思考中，请稍候...'):
            html_content, report_path = run_agent_pipeline(df, user_query, API_KEY, API_BASE)
            
        st.success("✅ 分析完成！")
        
        # 直接在 Web 页面渲染生成的 HTML 报告
        st.components.v1.html(html_content, height=800, scrolling=True)
        
        # 提供下载按钮
        with open(report_path, "r", encoding="utf-8") as f:
            st.download_button(
                label="📥 下载独立 HTML 报告",
                data=f.read(),
                file_name="AI_Analysis_Report.html",
                mime="text/html"
            )
else:
    st.info("👈 请先在左侧上传数据文件。")
