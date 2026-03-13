import os
import re
import base64
import markdown

def generate_html_report(markdown_text: str, output_path: str) -> str:
    print("   [->] 渲染引擎：开始扫描并替换图表占位符...")

    # ── Step 1：占位符格式容错归一化 ─────────────────────────────────────────
    # 覆盖所有 LLM 变体：[chart_1] / [CHART_1] / [CHART_1.png] /
    #                    [图表1] / [CHART_1: 折线图描述...] 等
    markdown_text = re.sub(
        r"\[(?:chart|图表|Chart|CHART)[_\s]*(\d+)(?:\.png)?(?:[：:][^\]]*)?\]",
        lambda m: f"[CHART_{m.group(1)}]",
        markdown_text,
        flags=re.IGNORECASE,
    )

    # ── Step 2：先做 Markdown → HTML 转换 ────────────────────────────────────
    # 必须在替换 <img> 之前转换！否则 markdown 库会把 <img> 里的引号转义成 &quot;
    print("   [->] 渲染引擎：开始进行 Markdown 到 HTML 的格式转换...")
    try:
        html_body = markdown.markdown(markdown_text, extensions=["tables", "fenced_code"])
        print("   [+] Markdown 转换顺利完成！")
    except Exception as e:
        print(f"   [-] Markdown 解析报错 ({e})，启用备用纯文本渲染！")
        html_body = f"<pre style='white-space: pre-wrap; font-family: inherit;'>{markdown_text}</pre>"

    # ── Step 3：在 HTML 中替换占位符为 Base64 内联图片 ───────────────────────
    # 此时占位符已经过 markdown 转换，需匹配 markdown 可能产生的包装形式
    # markdown 库会把 [CHART_1] 渲染为纯文本 [CHART_1]（不是链接），可直接替换
    print("   [->] 渲染引擎：开始嵌入图表图片...")
    for i in range(1, 10):
        chart_filename = f"chart_{i}.png"
        placeholder = f"[CHART_{i}]"  # markdown 输出里保留原文

        if placeholder in html_body:
            if os.path.exists(chart_filename):
                try:
                    with open(chart_filename, "rb") as f:
                        b64 = base64.b64encode(f.read()).decode("utf-8")
                    img_html = (
                        f'<figure>'
                        f'<img src="data:image/png;base64,{b64}" '
                        f'class="report-img" alt="分析图表 {i}">'
                        f'</figure>'
                    )
                    html_body = html_body.replace(placeholder, img_html)
                    os.remove(chart_filename)
                    print(f"   [+] 成功将 {chart_filename} 嵌入报告！")
                except Exception as e:
                    print(f"   [-] 图片 {chart_filename} 嵌入失败: {e}")
            else:
                print(f"   [!] 警告：AI 虚构了 {placeholder}，已执行降级拦截。")
                html_body = html_body.replace(
                    placeholder,
                    f"<p style='color:#e74c3c;'><i>[系统拦截：此处图表 {i} 未实际生成]</i></p>",
                )

    # ── Step 4：组装最终 HTML ─────────────────────────────────────────────────
    html_content = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <title>AI 智能数据洞察报告</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; background-color: #f4f7f6; color: #333; line-height: 1.6; }}
        .container {{ max-width: 900px; margin: 0 auto; background: #fff; padding: 40px; border-radius: 8px; box-shadow: 0 4px 15px rgba(0,0,0,0.05); }}
        h1 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 15px; text-align: center; margin-bottom: 30px; }}
        h2, h3 {{ color: #2980b9; margin-top: 25px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        th {{ background-color: #f8f9fa; color: #2c3e50; }}
        .report-img {{ max-width: 100%; height: auto; display: block; margin: 20px auto; border: 1px solid #eee; border-radius: 6px; box-shadow: 0 2px 8px rgba(0,0,0,0.05); }}
        figure {{ margin: 20px 0; text-align: center; }}
        .footer {{ margin-top: 40px; text-align: center; color: #95a5a6; font-size: 13px; border-top: 1px solid #eee; padding-top: 20px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 AI 智能多维数据洞察报告</h1>
        <div class="content-box">
            {html_body}
        </div>
        <div class="footer">
            生成引擎：Multi-Agent 架构 &bull; 全流式解析版
        </div>
    </div>
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print("[+] 🎉 HTML 报告打包完毕！\n===========================================")
    return html_content
