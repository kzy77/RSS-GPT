# 确保正确设置API密钥
def check_api_access():
    try:
        # 验证API访问权限
        response = api.validate_access()
        return response.is_valid
    except Exception as e:
        logging.error(f"API访问验证失败: {str(e)}")
        return False 