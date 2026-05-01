@echo off
REM =============================================================================
REM LangChain 示例依赖安装脚本
REM =============================================================================
REM
REM 用途：一键安装 LangChain 教程所需依赖
REM =============================================================================

echo.
echo ==============================================================================
echo   LangChain 教程 - 依赖安装脚本
echo ==============================================================================
echo.

echo [1/3] 升级 pip...
python -m pip install --upgrade pip

echo.
echo [2/3] 安装核心依赖...
pip install langchain langchain-community langchain-ollama python-dotenv pydantic

echo.
echo [3/3] 安装可选依赖（用于 RAG 检索）...
pip install langchain-huggingface faiss-cpu langchain-openai

echo.
echo ==============================================================================
echo   安装完成！
echo ==============================================================================
echo.
echo 下一步：
echo   1. 如果使用本地模型：安装 Ollama (https://ollama.ai)
echo   2. 下载模型：ollama pull qwen3.5:2b
echo   3. 运行示例：python 01_introduction/first_chain.py
echo.
echo 如遇问题：
echo   - 检查 Python 版本（建议 3.8+）
echo   - 检查网络连接
echo   - 使用国内镜像：pip install -i https://pypi.tuna.tsinghua.edu.cn/simple ...
echo.
