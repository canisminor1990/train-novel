# 在项目根目录运行。没有 pixi 时先装 pixi，再按 pixi.lock 装好 Python 环境（Unsloth、CUDA 版 PyTorch）。
# 依赖写在 pyproject.toml，版本锁在 pixi.lock。改依赖用 pixi add，不要在环境里直接 pip install。
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

# 这台机器上的 Python 不能在用户 AppData 里建文件，pixi 构建本项目时会失败，所以缓存放在仓库的 .cache 里。
$env:PIXI_CACHE_DIR = Join-Path $PSScriptRoot ".cache\pixi"
$pixiBin = Join-Path $env:USERPROFILE ".pixi\bin"
if (-not (Get-Command pixi -ErrorAction SilentlyContinue)) {
    if (-not (Test-Path (Join-Path $pixiBin "pixi.exe"))) {
        Write-Host "没找到 pixi，从 pixi.sh 安装"
        Invoke-RestMethod https://pixi.sh/install.ps1 | Invoke-Expression
    }
    $env:PATH = "$pixiBin;$env:PATH"
}
if (-not (Test-Path .\.env)) {
    Copy-Item .\.env.example .\.env
}

& pixi install --locked
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& pixi run --locked check-env
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Write-Host "环境就绪。双击 start.bat 打开训练台。"
