try:
    summary = generate_summary(article)
except Exception as e:
    # 记录错误
    logging.error(f"摘要生成失败: {str(e)}")
    # 使用备用方案
    summary = article[:500] + "..."  # 简单截取作为备用方案 