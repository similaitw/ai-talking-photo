[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)
$OutputEncoding = [Console]::OutputEncoding

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$VenvDir = Join-Path $ProjectRoot ".venv"
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"
$Wav2LipDir = Join-Path $ProjectRoot "vendor\Wav2Lip"
$Wav2LipRevision = "bac9a81e63ecc153202353372e5724b83d9e6322"
$ModelsDir = Join-Path $ProjectRoot "models"
$Checkpoint = Join-Path $ModelsDir "wav2lip_gan.pth"
$OfficialGanCopy = Join-Path $ModelsDir "wav2lip_gan.official.torchscript"
$Detector = Join-Path $Wav2LipDir "face_detection\detection\sfd\s3fd.pth"
$GanFileId = "15G3U08c8xsCkOqQxE38Z2XXDnPcOptNk"
$S3fdUrl = "https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth"

function Write-Step {
    param([string]$Message)
    Write-Host "`n==> $Message" -ForegroundColor Cyan
}

function Get-CommandPath {
    param([string]$Name)
    $command = Get-Command $Name -ErrorAction SilentlyContinue
    if ($null -eq $command) {
        return $null
    }
    return $command.Source
}

function Invoke-Native {
    param(
        [string]$FilePath,
        [string[]]$Arguments,
        [string]$FailureMessage
    )
    & $FilePath @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$FailureMessage（結束代碼：$LASTEXITCODE）"
    }
}

function Invoke-NativeOutput {
    param(
        [string]$FilePath,
        [string[]]$Arguments,
        [string]$FailureMessage
    )
    $result = & $FilePath @Arguments 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw "$FailureMessage（結束代碼：$LASTEXITCODE）"
    }
    return (($result | Out-String).Trim())
}

function Resolve-Python311 {
    $pyLauncher = Get-CommandPath "py.exe"
    if ($null -eq $pyLauncher) {
        $pyLauncher = Get-CommandPath "py"
    }
    if ($null -ne $pyLauncher) {
        try {
            $version = Invoke-NativeOutput $pyLauncher @("-3.11", "-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')") "無法啟動 Python 3.11"
            if ($version -eq "3.11") {
                return [PSCustomObject]@{ FilePath = $pyLauncher; Prefix = @("-3.11") }
            }
        }
        catch {
            # 後續仍嘗試 PATH 中的 python.exe。
        }
    }

    $python = Get-CommandPath "python.exe"
    if ($null -eq $python) {
        $python = Get-CommandPath "python"
    }
    if ($null -ne $python) {
        try {
            $version = Invoke-NativeOutput $python @("-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')") "無法確認 Python 版本"
            if ($version -eq "3.11") {
                return [PSCustomObject]@{ FilePath = $python; Prefix = @() }
            }
        }
        catch {
            # 由下方統一顯示安裝提示。
        }
    }

    throw "找不到 Python 3.11。請先安裝 64 位元 Python 3.11，並重新開啟 PowerShell；本腳本不會自動安裝 Python。"
}

$OriginalLocation = Get-Location
try {
    Set-Location $ProjectRoot
    Write-Host "AI Talking Photo — Windows 環境設定" -ForegroundColor Green
    Write-Host "專案位置：$ProjectRoot"

    Write-Step "檢查 Git、Python 3.11 與 FFmpeg"
    $Git = Get-CommandPath "git.exe"
    if ($null -eq $Git) {
        $Git = Get-CommandPath "git"
    }
    if ($null -eq $Git) {
        throw "找不到 Git。請先安裝 Git for Windows，並確認 git 指令可在 PowerShell 執行。"
    }
    Invoke-Native $Git @("--version") "Git 無法執行"

    $PythonLauncher = Resolve-Python311
    Write-Host "已找到 Python 3.11：$($PythonLauncher.FilePath)"

    $FFmpeg = Get-CommandPath "ffmpeg.exe"
    if ($null -eq $FFmpeg) {
        $FFmpeg = Get-CommandPath "ffmpeg"
    }
    if ($null -eq $FFmpeg) {
        throw "找不到 FFmpeg。請先安裝 FFmpeg 並加入 PATH，再重新執行此腳本。"
    }
    Invoke-Native $FFmpeg @("-version") "FFmpeg 無法執行"

    Write-Step "檢查 NVIDIA GPU／驅動程式"
    $NvidiaSmi = Get-CommandPath "nvidia-smi.exe"
    if ($null -eq $NvidiaSmi) {
        $NvidiaSmi = Get-CommandPath "nvidia-smi"
    }
    if ($null -eq $NvidiaSmi) {
        Write-Warning "找不到 nvidia-smi；本腳本不會安裝或更新 NVIDIA 驅動程式。仍可完成環境設定，但本機 CUDA 可能不可用。"
    }
    else {
        Invoke-Native $NvidiaSmi @("--query-gpu=name,memory.total,driver_version", "--format=csv,noheader") "無法讀取 NVIDIA GPU 資訊"
    }

    Write-Step "建立或確認 Python 3.11 虛擬環境"
    if (Test-Path $VenvPython) {
        $venvVersion = Invoke-NativeOutput $VenvPython @("-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')") "既有 .venv 無法執行"
        if ($venvVersion -ne "3.11") {
            throw "既有 .venv 不是 Python 3.11（目前為 $venvVersion）。為避免破壞既有環境，請先手動移走或刪除 .venv 後再執行。"
        }
        Write-Host "沿用既有 Python 3.11 .venv。"
    }
    else {
        Invoke-Native $PythonLauncher.FilePath ($PythonLauncher.Prefix + @("-m", "venv", $VenvDir)) "建立 .venv 失敗"
    }

    Write-Step "安裝已驗證的 PyTorch 與 Wav2Lip 相依套件"
    Invoke-Native $VenvPython @("-m", "pip", "install", "--upgrade", "pip") "更新 pip 失敗"
    Invoke-Native $VenvPython @("-m", "pip", "install", "torch==2.5.1", "--index-url", "https://download.pytorch.org/whl/cu118") "安裝 PyTorch 2.5.1+cu118 失敗"
    Invoke-Native $VenvPython @("-m", "pip", "install", "-r", (Join-Path $ProjectRoot "requirements-wav2lip-lock.txt")) "安裝 Wav2Lip 相依套件失敗"
    $TorchInfo = Invoke-NativeOutput $VenvPython @("-c", "import torch; print(f'PyTorch {torch.__version__} | CUDA runtime {torch.version.cuda} | CUDA available {torch.cuda.is_available()}')") "無法檢查 PyTorch／CUDA"
    Write-Host $TorchInfo

    Write-Step "準備官方 Wav2Lip"
    $VendorDir = Split-Path $Wav2LipDir -Parent
    New-Item -ItemType Directory -Force -Path $VendorDir | Out-Null
    if (-not (Test-Path $Wav2LipDir)) {
        Invoke-Native $Git @("clone", "https://github.com/Rudrabha/Wav2Lip.git", $Wav2LipDir) "下載官方 Wav2Lip 失敗"
    }
    elseif (-not (Test-Path (Join-Path $Wav2LipDir ".git"))) {
        throw "vendor\Wav2Lip 已存在但不是 Git repository。為避免覆蓋資料，請先手動移走該資料夾。"
    }
    Invoke-Native $Git @("-C", $Wav2LipDir, "fetch", "--depth", "1", "origin", $Wav2LipRevision) "更新官方 Wav2Lip 版本失敗"
    Invoke-Native $Git @("-C", $Wav2LipDir, "checkout", "--force", $Wav2LipRevision) "切換官方 Wav2Lip 版本失敗"

    Write-Step "準備官方模型權重"
    New-Item -ItemType Directory -Force -Path $ModelsDir | Out-Null
    if (-not (Test-Path $Checkpoint) -and -not (Test-Path $OfficialGanCopy)) {
        Invoke-Native $VenvPython @("-m", "gdown", $GanFileId, "-O", $Checkpoint) "下載官方 Wav2Lip GAN 權重失敗"
    }
    else {
        Write-Host "GAN 權重已存在，略過重複下載。"
    }

    $DetectorDir = Split-Path $Detector -Parent
    New-Item -ItemType Directory -Force -Path $DetectorDir | Out-Null
    if (-not (Test-Path $Detector)) {
        try {
            Invoke-WebRequest -Uri $S3fdUrl -OutFile $Detector -UseBasicParsing
        }
        catch {
            Remove-Item $Detector -Force -ErrorAction SilentlyContinue
            throw "下載官方 S3FD 人臉偵測權重失敗：$($_.Exception.Message)"
        }
    }
    else {
        Write-Host "S3FD 權重已存在，略過重複下載。"
    }

    Write-Step "驗證並準備 Wav2Lip 相容格式"
    Invoke-Native $VenvPython @((Join-Path $ProjectRoot "scripts\prepare_wav2lip.py")) "Wav2Lip 權重驗證／相容性準備失敗"

    Write-Step "執行 Doctor"
    Invoke-Native $VenvPython @((Join-Path $ProjectRoot "scripts\doctor.py")) "Doctor 檢查未通過"

    Write-Host "`nWindows 環境設定完成。" -ForegroundColor Green
    Write-Host "下一步："
    Write-Host "  .\.venv\Scripts\Activate.ps1"
    Write-Host "  python scripts\doctor.py"
    Write-Host "  python app.py"
    Write-Host "啟動後預設網址：http://127.0.0.1:7860"
    Write-Host "GTX 1050 2GB 建議優先使用 Colab；本機低顯示記憶體模式留待 M6.2 進一步驗證。"
}
finally {
    Set-Location $OriginalLocation
}
