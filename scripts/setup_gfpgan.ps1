[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)
$OutputEncoding = [Console]::OutputEncoding

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$VenvDir = Join-Path $ProjectRoot ".venv-gfpgan"
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"
$MainVenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$GFPGANDir = Join-Path $ProjectRoot "vendor\GFPGAN"
$GFPGANRevision = "7552a7791caad982045a7bbe5634bbf1cd5c8679"
$ModelDir = Join-Path $GFPGANDir "gfpgan\weights"
$ModelPath = Join-Path $ModelDir "GFPGANv1.3.pth"
$ModelUrl = "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth"

function Write-Step {
    param([string]$Message)
    Write-Host "`n==> $Message" -ForegroundColor Cyan
}

function Get-CommandPath {
    param([string]$Name)
    $command = Get-Command $Name -ErrorAction SilentlyContinue
    if ($null -eq $command) { return $null }
    return $command.Source
}

function Invoke-Native {
    param([string]$FilePath, [string[]]$Arguments, [string]$FailureMessage)

    # Windows PowerShell 5.1 represents native stderr as ErrorRecord objects.
    # Temporarily avoid Stop semantics and judge failure only by LASTEXITCODE.
    $previousErrorActionPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        & $FilePath @Arguments
        $exitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previousErrorActionPreference
    }

    if ($exitCode -ne 0) {
        throw "$FailureMessage（結束代碼：$exitCode）"
    }
}

function Invoke-NativeOutput {
    param([string]$FilePath, [string[]]$Arguments, [string]$FailureMessage)

    # PowerShell 5.1 can promote native stderr to NativeCommandError even when
    # stderr is redirected. Run the native command with Continue semantics,
    # capture stderr separately, then judge success only from LASTEXITCODE.
    $stderrPath = [System.IO.Path]::GetTempFileName()
    $previousErrorActionPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        $result = & $FilePath @Arguments 2> $stderrPath
        $exitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previousErrorActionPreference
    }

    try {
        $stderrText = ""
        if (Test-Path $stderrPath) {
            $stderrText = (Get-Content $stderrPath -Raw -ErrorAction SilentlyContinue)
        }

        if ($exitCode -ne 0) {
            if (-not [string]::IsNullOrWhiteSpace($stderrText)) {
                Write-Host $stderrText -ForegroundColor Red
            }
            throw "$FailureMessage（結束代碼：$exitCode）"
        }

        if (-not [string]::IsNullOrWhiteSpace($stderrText)) {
            Write-Host $stderrText -ForegroundColor DarkYellow
        }
        return (($result | Out-String).Trim())
    }
    finally {
        Remove-Item $stderrPath -Force -ErrorAction SilentlyContinue
    }
}

function Resolve-Python311 {
    $py = Get-CommandPath "py.exe"
    if ($null -eq $py) { $py = Get-CommandPath "py" }
    if ($null -ne $py) {
        try {
            $version = Invoke-NativeOutput $py @("-3.11", "-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')") "無法啟動 Python 3.11"
            if ($version -eq "3.11") {
                return [PSCustomObject]@{ FilePath = $py; Prefix = @("-3.11") }
            }
        } catch {}
    }
    $python = Get-CommandPath "python.exe"
    if ($null -eq $python) { $python = Get-CommandPath "python" }
    if ($null -ne $python) {
        try {
            $version = Invoke-NativeOutput $python @("-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')") "無法確認 Python 版本"
            if ($version -eq "3.11") {
                return [PSCustomObject]@{ FilePath = $python; Prefix = @() }
            }
        } catch {}
    }
    throw "找不到 Python 3.11。請先安裝 64 位元 Python 3.11。"
}

$OriginalLocation = Get-Location
try {
    Set-Location $ProjectRoot
    Write-Host "AI Talking Photo — GFPGAN 高清修復設定" -ForegroundColor Green
    Write-Host "專案位置：$ProjectRoot"

    Write-Step "檢查 Git、Python 3.11、主環境與 FFmpeg"
    $Git = Get-CommandPath "git.exe"
    if ($null -eq $Git) { $Git = Get-CommandPath "git" }
    if ($null -eq $Git) { throw "找不到 Git。請先安裝 Git for Windows。" }
    $PythonLauncher = Resolve-Python311
    if (-not (Test-Path $MainVenvPython)) {
        throw "找不到主 .venv。請先執行 scripts/setup_windows.ps1 完成 Wav2Lip 主環境設定。"
    }
    $FFmpeg = Get-CommandPath "ffmpeg.exe"
    if ($null -eq $FFmpeg) { $FFmpeg = Get-CommandPath "ffmpeg" }
    if ($null -eq $FFmpeg) { throw "找不到 FFmpeg。請先安裝 FFmpeg 並加入 PATH。" }

    Write-Step "建立或確認獨立 .venv-gfpgan"
    if (Test-Path $VenvPython) {
        $version = Invoke-NativeOutput $VenvPython @("-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')") "既有 .venv-gfpgan 無法執行"
        if ($version -ne "3.11") {
            throw "既有 .venv-gfpgan 不是 Python 3.11，請先手動移走後重試。"
        }
        Write-Host "沿用既有 .venv-gfpgan。"
    } else {
        Invoke-Native $PythonLauncher.FilePath ($PythonLauncher.Prefix + @("-m", "venv", $VenvDir)) "建立 .venv-gfpgan 失敗"
    }

    Write-Step "安裝 GFPGAN 專用 PyTorch／torchvision"
    Invoke-Native $VenvPython @("-m", "pip", "install", "--upgrade", "pip") "更新 pip 失敗"
    Invoke-Native $VenvPython @("-m", "pip", "install", "torch==2.1.2", "torchvision==0.16.2", "--index-url", "https://download.pytorch.org/whl/cu118") "安裝 GFPGAN PyTorch 環境失敗"
    Invoke-Native $VenvPython @("-m", "pip", "install", "-r", (Join-Path $ProjectRoot "requirements-gfpgan-lock.txt")) "安裝 GFPGAN 相依套件失敗"

    Write-Step "準備官方 TencentARC/GFPGAN"
    $VendorDir = Split-Path $GFPGANDir -Parent
    New-Item -ItemType Directory -Force -Path $VendorDir | Out-Null
    if (-not (Test-Path $GFPGANDir)) {
        Invoke-Native $Git @("clone", "https://github.com/TencentARC/GFPGAN.git", $GFPGANDir) "下載官方 GFPGAN 失敗"
    } elseif (-not (Test-Path (Join-Path $GFPGANDir ".git"))) {
        throw "vendor\GFPGAN 已存在但不是 Git repository。為避免覆蓋資料，請先手動移走。"
    }
    Invoke-Native $Git @("-C", $GFPGANDir, "fetch", "--depth", "1", "origin", $GFPGANRevision) "更新官方 GFPGAN 版本失敗"
    Invoke-Native $Git @("-C", $GFPGANDir, "checkout", "--force", $GFPGANRevision) "切換官方 GFPGAN 版本失敗"
    Invoke-Native $VenvPython @("-m", "pip", "install", "-e", $GFPGANDir, "--no-deps") "安裝 GFPGAN 程式失敗"

    Write-Step "準備 GFPGAN V1.3 官方權重"
    New-Item -ItemType Directory -Force -Path $ModelDir | Out-Null
    if (-not (Test-Path $ModelPath)) {
        try {
            Invoke-WebRequest -Uri $ModelUrl -OutFile $ModelPath -UseBasicParsing
        } catch {
            Remove-Item $ModelPath -Force -ErrorAction SilentlyContinue
            throw "下載 GFPGAN V1.3 權重失敗：$($_.Exception.Message)"
        }
    } else {
        Write-Host "GFPGAN V1.3 權重已存在，略過重複下載。"
    }

    Write-Step "驗證 GFPGAN 執行環境"
    $Info = Invoke-NativeOutput $VenvPython @("-c", "import torch, gfpgan; print(f'PyTorch {torch.__version__} | CUDA available {torch.cuda.is_available()} | GFPGAN import OK')") "GFPGAN 環境驗證失敗"
    Write-Host $Info

    Write-Step "重新套用 Wav2Lip 相容性與嘴周柔和融合修正"
    Invoke-Native $MainVenvPython @((Join-Path $ProjectRoot "scripts\prepare_wav2lip.py")) "Wav2Lip 品質修正準備失敗"

    Write-Host "`nGFPGAN 高清修復設定完成。" -ForegroundColor Green
    Write-Host "請用原本 .venv 啟動 app.py；主程式會自動呼叫 .venv-gfpgan。"
    Write-Host "Gradio 中將『畫質後處理』選成『GFPGAN 高清修復（實驗）』即可 A/B 測試。"
    Write-Host "GTX 1050 2GB 預設只做 1× 中心人臉修復，不啟用 Real-ESRGAN 背景放大。"
}
finally {
    Set-Location $OriginalLocation
}
