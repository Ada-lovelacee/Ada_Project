if (!(Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Host "未检测到 Python。请先安装 Python 3.11+ 并勾选 Add python.exe to PATH。" -ForegroundColor Yellow
    Write-Host "下载地址: https://www.python.org/downloads/" -ForegroundColor Cyan
    exit 1
}

if (!(Get-Command docker -ErrorAction SilentlyContinue)) {
    Write-Host "未检测到 Docker。建议安装 Docker Desktop 来快速启动 MySQL。" -ForegroundColor Yellow
    Write-Host "下载地址: https://www.docker.com/products/docker-desktop/" -ForegroundColor Cyan
    Write-Host "如果你准备本机安装 MySQL，也可以跳过 Docker，手动创建数据库后再运行 Python 部分命令。" -ForegroundColor Yellow
    exit 1
}

if (!(Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
}

docker compose up -d

if (!(Test-Path ".venv")) {
    python -m venv .venv
}

& ".\.venv\Scripts\python.exe" -m pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple
& ".\.venv\Scripts\python.exe" -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
& ".\.venv\Scripts\python.exe" run.py
