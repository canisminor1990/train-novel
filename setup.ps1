# 在项目根目录运行。已有 .venv 时会接着装，不会重装 Python。
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

function Get-TorchIndexes([string]$tag) {
    return @(
        "https://mirrors.nju.edu.cn/pytorch/whl/$tag",
        "https://mirror.sjtu.edu.cn/pytorch-wheels/$tag",
        "https://download.pytorch.org/whl/$tag"
    )
}

function Test-WheelIndex([string]$index) {
    try {
        $html = (Invoke-WebRequest -Uri "$index/torch/" -UseBasicParsing -TimeoutSec 15).Content
        return $html -match "cp312-cp312-win_amd64"
    } catch {
        return $false
    }
}

function Get-CudaCandidates {
    $tags = @()
    $smi = & nvidia-smi | Out-String
    if ($smi -match "CUDA UMD Version:\s*([0-9]+)\.([0-9]+)") {
        $tags += "cu{0}{1}" -f $Matches[1], $Matches[2]
    } elseif ($smi -match "CUDA Version:\s*([0-9]+)\.([0-9]+)") {
        $tags += "cu{0}{1}" -f $Matches[1], $Matches[2]
    }
    $tags += @("cu132", "cu130", "cu128", "cu126")
    return $tags | Select-Object -Unique
}

if (-not (Test-Path .\.venv\Scripts\python.exe)) {
    py -3.12 -m venv .venv
}
if (-not (Test-Path .\.env)) {
    Copy-Item .\.env.example .\.env
}

$python = ".\.venv\Scripts\python.exe"
& $python -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $python -m pip install unsloth
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $python -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)"
if ($LASTEXITCODE -ne 0) {
    $chosen = $null
    foreach ($tag in Get-CudaCandidates) {
        foreach ($index in Get-TorchIndexes $tag) {
            if (Test-WheelIndex $index) {
                $chosen = $index
                break
            }
        }
        if ($chosen) { break }
    }
    if (-not $chosen) {
        Write-Host "没有找到适用于 Windows + Python 3.12 的 CUDA 版 PyTorch。"
        exit 1
    }
    Write-Host "当前 torch 看不到 GPU，从国内镜像安装：$chosen"
    & $python -m pip install --upgrade --force-reinstall torch torchvision --index-url $chosen
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    & $python -m pip install "fsspec==2026.2.0" -i https://pypi.tuna.tsinghua.edu.cn/simple
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

& $python -m novel_train.check
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Write-Host "环境就绪。双击 start.bat 打开训练台。"
