# 当前开发环境（2026-10-04）

本机已创建仓库专用 .venv，实际 Python 为 3.12.14。本次验证使用
requirements-autoresearch.lock.txt 中的离线研究依赖，Node 依赖锁定在
package-lock.json，安装于项目 node_modules/。

```powershell
& .\.venv\Scripts\python.exe --version
& .\.venv\Scripts\python.exe -m pip check
& .\.venv\Scripts\python.exe -m pytest tests/physio_autoresearch --basetemp runs/physio-tests -q
npm run test:pi
```

这次 .venv 不包含历史文档曾声称安装的完整 Torch/Gradio/Jupyter 环境。
requirements.txt、requirements-dev.txt、requirements-legacy.txt 和根目录
.python-version 的历史声明没有用于本轮安装，也没有被当成验证通过的证据。
后续开发 representation 或旧 Demo 时，需要另行验证它们的依赖与 Python 兼容性。

重建本轮环境时，用可运行的 Python 3.12 创建 .venv，然后执行：

```powershell
# 本机全局 pip.ini 无法解析；这个覆盖仅影响当前 PowerShell 安装进程。
$env:PIP_CONFIG_FILE = 'NUL'
& .\.venv\Scripts\python.exe -m pip install -r requirements-autoresearch.lock.txt
npm ci --ignore-scripts
```

没有修改全局 pip 配置、安装系统 Python、登录模型供应商或迁移凭据。
详细工作流见 [physio_autoresearch/README.md](physio_autoresearch/README.md)。
