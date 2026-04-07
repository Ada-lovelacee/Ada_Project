# 联邦学习可视化系统

一个可以快速启动的全栈演示项目：

- 后端：Flask + SQLAlchemy + MySQL
- 前端：Bootstrap 5 + Chart.js
- 功能：联邦训练任务总览、客户端节点状态、全局模型收敛曲线、模拟下一轮训练

## 1. 项目结构

```text
ProjectDemo/
├─ app/
│  ├─ static/
│  ├─ templates/
│  ├─ __init__.py
│  ├─ config.py
│  ├─ extensions.py
│  ├─ models.py
│  ├─ routes.py
│  └─ seed.py
├─ .env.example
├─ docker-compose.yml
├─ requirements.txt
└─ run.py
```

## 2. 最快启动方式

你当前这台机器如果和我检测到的一样，缺的是：

- `Python`
- `Docker`

所以最推荐的顺序是先装这两个，再执行项目脚本。

### 先装什么

1. 安装 Python 3.11 或 3.12

下载地址：
[https://www.python.org/downloads/](https://www.python.org/downloads/)

安装时务必勾选：

- `Add python.exe to PATH`

2. 安装 Docker Desktop

下载地址：
[https://www.docker.com/products/docker-desktop/](https://www.docker.com/products/docker-desktop/)

安装完成后重启电脑一次更稳妥。

### 方案 A：用 Docker 启动 MySQL，最省事

前提：先安装 [Docker Desktop](https://www.docker.com/products/docker-desktop/)

也可以直接执行一键脚本：

```powershell
.\start.ps1
```

这个脚本会自动：

- 复制 `.env.example` 为 `.env`
- 启动 MySQL 容器
- 创建 Python 虚拟环境
- 安装依赖
- 启动 Flask

1. 启动 MySQL

```powershell
docker compose up -d
```

2. 创建 Python 虚拟环境

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

3. 安装依赖

```powershell
pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

4. 复制环境变量文件

```powershell
Copy-Item .env.example .env
```

5. 启动 Flask

```powershell
python run.py
```

6. 打开浏览器

```text
http://127.0.0.1:5000
```

### 方案 B：本机安装 MySQL

如果你不想装 Docker，可以安装 MySQL 8.x，然后：

1. 新建数据库

```sql
CREATE DATABASE federated_viz DEFAULT CHARSET utf8mb4;
```

2. 修改 `.env` 里的账号密码
3. 执行：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
python run.py
```

## 额外提速

如果你希望以后 `pip install` 默认都走国内镜像，可以执行：

```powershell
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

这样后续就不需要每次手动加 `-i` 参数了。

应用首次启动时会自动建表并写入演示数据。

## 3. 页面能力

- 首页展示联邦学习系统简介和快速操作按钮
- 仪表盘展示全局准确率、在线节点数、平均通信时延
- 折线图展示每轮聚合后的准确率和损失变化
- 表格展示客户端节点状态、数据量、贡献度
- 点击“模拟下一轮训练”可生成新一轮联邦训练结果

## 4. API

### `GET /api/overview`

返回任务概览、客户端列表和轮次数据。

### `POST /api/simulate`

模拟追加一轮联邦训练结果，并更新客户端状态。

## 5. 后续可以继续扩展

- 接入真实联邦学习框架结果，如 Flower、FedML、PaddleFL
- 增加用户登录、实验管理、模型版本管理
- 增加更多图表，例如客户端参与率、通信开销、异常节点告警
- 将前后端拆分为 Flask API + 独立前端项目
