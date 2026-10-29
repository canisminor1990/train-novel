# 一键启动训练台。双击 start.bat，或在 PowerShell 里运行 .\start.ps1
#   -Rebuild    强制重新构建前端
#   -NoBrowser  不自动打开浏览器
param(
    [switch]$Rebuild,
    [switch]$NoBrowser
)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
[Console]::OutputEncoding = [Text.Encoding]::UTF8
$env:PYTHONIOENCODING = "utf-8"

$Url = "http://127.0.0.1:8765"
$env:PIXI_CACHE_DIR = Join-Path $PSScriptRoot ".cache\pixi"
$PixiBin = Join-Path $env:USERPROFILE ".pixi\bin"
if (-not (Get-Command pixi -ErrorAction SilentlyContinue) -and (Test-Path (Join-Path $PixiBin "pixi.exe"))) {
    $env:PATH = "$PixiBin;$env:PATH"
}

function Step([string]$text) { Write-Host "`n▶ $text" -ForegroundColor Cyan }
function Fail([string]$text) { Write-Host "`n✖ $text" -ForegroundColor Red; exit 1 }

function Test-Http([string]$address) {
    try {
        Invoke-WebRequest -Uri $address -UseBasicParsing -TimeoutSec 2 | Out-Null
        return $true
    } catch {
        return $false
    }
}

function Open-Dashboard {
    if (-not $NoBrowser) { Start-Process $Url }
}

# 已经在跑：只打开页面
if (Test-Http "$Url/api/state") {
    Write-Host "训练台已经在运行：$Url"
    Open-Dashboard
    exit 0
}

# 1. Python 环境
if (-not (Get-Command pixi -ErrorAction SilentlyContinue) -or -not (Test-Path .\.pixi\envs\default\python.exe)) {
    Step "第一次运行，先安装 Python 环境（Unsloth、CUDA 版 PyTorch），需要几分钟到几十分钟"
    & powershell -NoProfile -ExecutionPolicy Bypass -File .\setup.ps1
    if ($LASTEXITCODE -ne 0) { Fail "环境安装没有完成，看上面的报错。" }
    if (-not (Get-Command pixi -ErrorAction SilentlyContinue)) { $env:PATH = "$PixiBin;$env:PATH" }
}

# 2. 前端
$dist = "web\dist\index.html"
$needBuild = $Rebuild -or -not (Test-Path $dist)
if (-not $needBuild) {
    $built = (Get-Item $dist).LastWriteTime
    # PowerShell 5.1 给文件路径加 -Recurse 时会搜遍上级目录（含 node_modules）找同名文件，会崩。
    $sources = @(Get-ChildItem web\src -Recurse -File) + @(Get-Item web\index.html, web\package.json)
    $changed = $sources | Where-Object { $_.LastWriteTime -gt $built } | Select-Object -First 1
    $needBuild = $null -ne $changed
}
if ($needBuild) {
    Step "构建训练台页面"
    if (-not (Get-Command node -ErrorAction SilentlyContinue)) {
        if (Test-Path $dist) {
            Write-Host "没装 Node.js，先用上次构建的页面。" -ForegroundColor Yellow
            $needBuild = $false
        } else {
            Fail "构建页面需要 Node.js 22 或更新版本：https://nodejs.org"
        }
    }
}
if ($needBuild) {
    if (-not (Get-Command pnpm -ErrorAction SilentlyContinue)) {
        Write-Host "没找到 pnpm，用 corepack 启用"
        & corepack enable pnpm
        if ($LASTEXITCODE -ne 0) { Fail "pnpm 启用失败。手动运行 npm install -g pnpm 后再试。" }
    }
    Push-Location web
    try {
        $env:CI = "true"
        if (-not (Test-Path node_modules)) {
            & pnpm install
            if ($LASTEXITCODE -ne 0) { Fail "前端依赖安装失败。" }
        }
        & pnpm build
        if ($LASTEXITCODE -ne 0) { Fail "页面构建失败。" }
    } finally {
        Remove-Item Env:CI -ErrorAction SilentlyContinue
        Pop-Location
    }
}

# 3. Ollama（挂载和续写用；没有也能训练）
if (-not (Test-Http "http://127.0.0.1:11434/api/version")) {
    $ollama = Get-Command ollama -ErrorAction SilentlyContinue
    $exe = if ($ollama) { $ollama.Source } else { Join-Path $env:LOCALAPPDATA "Programs\Ollama\ollama.exe" }
    if (Test-Path $exe) {
        Step "启动 Ollama"
        Start-Process -FilePath $exe -ArgumentList "serve" -WindowStyle Hidden
    } else {
        Write-Host "`n没装 Ollama：可以训练，但训完挂不上 Ollama，也不能在页面里续写。https://ollama.com/download" -ForegroundColor Yellow
    }
}

# 4. 训练台
Step "启动训练台 $Url"
Write-Host "关掉这个窗口或按 Ctrl+C 会停止训练台。已经开始的训练在后台继续，重新启动后接着显示。"
if (-not $NoBrowser) {
    Start-Job -ArgumentList $Url -ScriptBlock {
        param($address)
        for ($i = 0; $i -lt 60; $i++) {
            try {
                Invoke-WebRequest -Uri "$address/api/state" -UseBasicParsing -TimeoutSec 2 | Out-Null
                Start-Process $address
                return
            } catch {
                Start-Sleep -Milliseconds 500
            }
        }
    } | Out-Null
}
& pixi run --locked dashboard
exit $LASTEXITCODE
