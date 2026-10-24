@echo off
python -m pip install -r "%~dp0requirements.txt"
if errorlevel 1 exit /b 1
echo 安装完成。需要运行模型示例时，请先启动 Ollama 并下载 qwen3.5:2b。
