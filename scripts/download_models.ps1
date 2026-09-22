[CmdletBinding()]
param(
    [string]$ChecksumFile = "models/checksums.sha256",
    [switch]$InstallG2PW,
    [string]$G2PWArchivePath,
    [switch]$InstallV2ProSpeakerVector,
    [string]$V2ProSpeakerVectorPath,
    [switch]$InstallFastLangDetect,
    [string]$FastLangDetectPath,
    [switch]$InstallV2ProPlus,
    [string]$V2ProPlusPath
)

$ErrorActionPreference = "Stop"

function Get-Sha256Hex {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path
    )

    $hasher = [System.Security.Cryptography.SHA256]::Create()
    try {
        $stream = [System.IO.File]::OpenRead($Path)
        try {
            return [System.BitConverter]::ToString($hasher.ComputeHash($stream)).Replace("-", "").ToLowerInvariant()
        }
        finally {
            $stream.Dispose()
        }
    }
    finally {
        $hasher.Dispose()
    }
}

function Test-G2PWModelDirectory {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path
    )

    $required = @(
        "g2pW.onnx",
        "config.py",
        "POLYPHONIC_CHARS.txt",
        "MONOPHONIC_CHARS.txt",
        "bopomofo_to_pinyin_wo_tune_dict.json",
        "char_bopomofo_dict.json"
    )
    foreach ($name in $required) {
        if (-not (Test-Path -LiteralPath (Join-Path $Path $name) -PathType Leaf)) {
            throw "G2PW model is incomplete: $name"
        }
    }
}

$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$checksumPath = Join-Path $root $ChecksumFile
if (-not (Test-Path -LiteralPath $checksumPath)) {
    throw "Checksum manifest does not exist: $checksumPath"
}
$modelsRoot = (Resolve-Path (Join-Path $root "models")).Path
$modelsRootPrefix = $modelsRoot.TrimEnd([char[]]@('\', '/')) + [System.IO.Path]::DirectorySeparatorChar

if ($InstallV2ProPlus) {
    # GPT-SoVITS model object from the project's ModelScope model repository,
    # with official Hugging Face and its transport mirror as fallbacks. The
    # immutable SHA-256 below is published by the official Hugging Face repo.
    $v2ProPlusUrls = @(
        "https://www.modelscope.cn/models/XXXXRT/GPT-SoVITS-Pretrained/resolve/master/pretrained_models/v2Pro/s2Gv2ProPlus.pth",
        "https://huggingface.co/lj1995/GPT-SoVITS/resolve/main/v2Pro/s2Gv2ProPlus.pth?download=true",
        "https://hf-mirror.com/lj1995/GPT-SoVITS/resolve/main/v2Pro/s2Gv2ProPlus.pth?download=true"
    )
    $v2ProPlusSha256 = "d42a22bbbf65fb2bbdd45ad6a66841156977db45c7aabe0a6992ff378d9c7d3b"
    $v2ProPlusName = "s2Gv2ProPlus.pth"
    $v2ProPlusTarget = Join-Path $modelsRoot "gpt-sovits\v2Pro\$v2ProPlusName"
    $v2ProPlusDownloads = Join-Path $modelsRoot "downloads"
    $v2ProPlusPart = Join-Path $v2ProPlusDownloads "$v2ProPlusName.part"

    if ($null -ne $V2ProPlusPath -and -not [string]::IsNullOrWhiteSpace($V2ProPlusPath)) {
        if ([System.IO.Path]::IsPathRooted($V2ProPlusPath)) {
            $v2ProPlusSource = [System.IO.Path]::GetFullPath($V2ProPlusPath)
        }
        else {
            $v2ProPlusSource = [System.IO.Path]::GetFullPath((Join-Path $root $V2ProPlusPath))
        }
        if (-not $v2ProPlusSource.StartsWith($modelsRootPrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
            throw "V2ProPlus generator must remain under models/: $v2ProPlusSource"
        }
    }
    elseif (Test-Path -LiteralPath $v2ProPlusTarget -PathType Leaf) {
        $v2ProPlusSource = $v2ProPlusTarget
    }
    else {
        New-Item -ItemType Directory -Force -Path $v2ProPlusDownloads | Out-Null
        $v2ProPlusSource = $v2ProPlusPart
        $downloaded = $false
        foreach ($url in $v2ProPlusUrls) {
            try {
                & curl.exe --ssl-no-revoke --fail --location --retry 3 --retry-all-errors --continue-at - --output $v2ProPlusSource $url
                if ($LASTEXITCODE -eq 0) {
                    $downloaded = $true
                    break
                }
            }
            catch {
                Write-Warning "V2ProPlus transport failed, trying the next fixed source: $url"
            }
        }
        if (-not $downloaded) {
            throw "V2ProPlus generator download failed from every fixed source"
        }
    }

    if (-not (Test-Path -LiteralPath $v2ProPlusSource -PathType Leaf)) {
        throw "V2ProPlus generator does not exist: $v2ProPlusSource"
    }
    if ((Get-Sha256Hex -Path $v2ProPlusSource) -ne $v2ProPlusSha256) {
        throw "V2ProPlus generator SHA-256 mismatch"
    }

    if ([System.IO.Path]::GetFullPath($v2ProPlusSource) -ne [System.IO.Path]::GetFullPath($v2ProPlusTarget)) {
        $targetParent = Split-Path -Parent $v2ProPlusTarget
        New-Item -ItemType Directory -Force -Path $targetParent | Out-Null
        $stagedTarget = "$v2ProPlusTarget.$([Guid]::NewGuid().ToString('N')).tmp"
        try {
            Copy-Item -LiteralPath $v2ProPlusSource -Destination $stagedTarget
            if ((Get-Sha256Hex -Path $stagedTarget) -ne $v2ProPlusSha256) {
                throw "Staged V2ProPlus generator SHA-256 mismatch"
            }
            Move-Item -LiteralPath $stagedTarget -Destination $v2ProPlusTarget -Force
        }
        finally {
            if (Test-Path -LiteralPath $stagedTarget -PathType Leaf) {
                Remove-Item -LiteralPath $stagedTarget -Force
            }
        }
    }
    if ((Get-Sha256Hex -Path $v2ProPlusTarget) -ne $v2ProPlusSha256) {
        throw "Deployed V2ProPlus generator SHA-256 mismatch"
    }
    if (Test-Path -LiteralPath $v2ProPlusPart -PathType Leaf) {
        Remove-Item -LiteralPath $v2ProPlusPart -Force
    }
    Write-Output "V2PROPLUS_GENERATOR_READY=$v2ProPlusTarget"
}

if ($InstallG2PW) {
    # This is the official archive URL used by the fixed GPT-SoVITS v2Pro source.
    # ModelScope currently returns this content digest in X-Linked-Etag.
    $g2pwUrl = "https://www.modelscope.cn/models/kamiorinn/g2pw/resolve/master/G2PWModel_1.1.zip"
    $g2pwArchiveSha256 = "b116f6930a7ee55eef6576a8d8e14bf40c1106583439e8ae924b901512379c64"
    $g2pwTarget = Join-Path $modelsRoot "gpt-sovits\GPT_SoVITS\text\G2PWModel"
    $g2pwTargetPrefix = $g2pwTarget.TrimEnd([char[]]@('\', '/')) + [System.IO.Path]::DirectorySeparatorChar

    if ($null -ne $G2PWArchivePath -and -not [string]::IsNullOrWhiteSpace($G2PWArchivePath)) {
        if ([System.IO.Path]::IsPathRooted($G2PWArchivePath)) {
            $archivePath = [System.IO.Path]::GetFullPath($G2PWArchivePath)
        }
        else {
            $archivePath = [System.IO.Path]::GetFullPath((Join-Path $root $G2PWArchivePath))
        }
        if (-not $archivePath.StartsWith($modelsRootPrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
            throw "G2PW archive must remain under models/: $archivePath"
        }
    }
    else {
        $downloadsDir = Join-Path $modelsRoot "downloads"
        New-Item -ItemType Directory -Force -Path $downloadsDir | Out-Null
        $archivePath = Join-Path $downloadsDir "G2PWModel_1.1.zip"
        if (-not (Test-Path -LiteralPath $archivePath -PathType Leaf)) {
            & curl.exe --ssl-no-revoke --fail --location --retry 3 --retry-all-errors --continue-at - --output $archivePath $g2pwUrl
            if ($LASTEXITCODE -ne 0) {
                throw "G2PW archive download failed with exit code $LASTEXITCODE"
            }
        }
    }

    if (-not (Test-Path -LiteralPath $archivePath -PathType Leaf)) {
        throw "G2PW archive does not exist: $archivePath"
    }
    if ((Get-Sha256Hex -Path $archivePath) -ne $g2pwArchiveSha256) {
        throw "G2PW archive SHA-256 mismatch"
    }

    if (-not (Test-Path -LiteralPath $g2pwTarget -PathType Container)) {
        $stagingRoot = Join-Path $modelsRoot ("downloads\g2pw-extract-" + [Guid]::NewGuid().ToString("N"))
        New-Item -ItemType Directory -Force -Path $stagingRoot | Out-Null
        Expand-Archive -LiteralPath $archivePath -DestinationPath $stagingRoot -Force
        # The checksum-verified official archive contains the model under
        # G2PWModel_1.1/; publish it locally under the stable vendor path.
        $stagedModel = Join-Path $stagingRoot "G2PWModel_1.1"
        Test-G2PWModelDirectory -Path $stagedModel
        $targetParent = Split-Path -Parent $g2pwTarget
        New-Item -ItemType Directory -Force -Path $targetParent | Out-Null
        Move-Item -LiteralPath $stagedModel -Destination $g2pwTarget
    }

    Test-G2PWModelDirectory -Path $g2pwTarget
    Write-Output "G2PW_MODEL_READY=$g2pwTarget"
}

if ($InstallV2ProSpeakerVector) {
    # Official ModelScope mirror of the GPT-SoVITS V2Pro ERes2NetV2 speaker-vector model.
    $speakerVectorUrl = "https://www.modelscope.cn/models/iic/speech_eres2netv2w24s4ep4_sv_zh-cn_16k-common/resolve/master/pretrained_eres2netv2w24s4ep4.ckpt"
    $speakerVectorSha256 = "740bb6584a99ee4cf910101536acba38c15a8017ea6a3a2813ec668fb62981f1"
    $speakerVectorName = "pretrained_eres2netv2w24s4ep4.ckpt"
    $speakerVectorTarget = Join-Path $modelsRoot "gpt-sovits\sv\$speakerVectorName"

    if ($null -ne $V2ProSpeakerVectorPath -and -not [string]::IsNullOrWhiteSpace($V2ProSpeakerVectorPath)) {
        if ([System.IO.Path]::IsPathRooted($V2ProSpeakerVectorPath)) {
            $speakerVectorSource = [System.IO.Path]::GetFullPath($V2ProSpeakerVectorPath)
        }
        else {
            $speakerVectorSource = [System.IO.Path]::GetFullPath((Join-Path $root $V2ProSpeakerVectorPath))
        }
        if (-not $speakerVectorSource.StartsWith($modelsRootPrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
            throw "V2Pro speaker-vector model must remain under models/: $speakerVectorSource"
        }
    }
    else {
        $downloadsDir = Join-Path $modelsRoot "downloads"
        New-Item -ItemType Directory -Force -Path $downloadsDir | Out-Null
        $speakerVectorSource = Join-Path $downloadsDir $speakerVectorName
        if (-not (Test-Path -LiteralPath $speakerVectorSource -PathType Leaf)) {
            & curl.exe --ssl-no-revoke --fail --location --retry 3 --retry-all-errors --continue-at - --output $speakerVectorSource $speakerVectorUrl
            if ($LASTEXITCODE -ne 0) {
                throw "V2Pro speaker-vector model download failed with exit code $LASTEXITCODE"
            }
        }
    }

    if (-not (Test-Path -LiteralPath $speakerVectorSource -PathType Leaf)) {
        throw "V2Pro speaker-vector model does not exist: $speakerVectorSource"
    }
    if ((Get-Sha256Hex -Path $speakerVectorSource) -ne $speakerVectorSha256) {
        throw "V2Pro speaker-vector model SHA-256 mismatch"
    }

    if (-not (Test-Path -LiteralPath $speakerVectorTarget -PathType Leaf)) {
        $targetParent = Split-Path -Parent $speakerVectorTarget
        New-Item -ItemType Directory -Force -Path $targetParent | Out-Null
        if ([System.IO.Path]::GetFullPath($speakerVectorSource) -ne [System.IO.Path]::GetFullPath($speakerVectorTarget)) {
            Copy-Item -LiteralPath $speakerVectorSource -Destination $speakerVectorTarget
        }
    }
    if (-not (Test-Path -LiteralPath $speakerVectorTarget -PathType Leaf)) {
        throw "V2Pro speaker-vector model was not deployed: $speakerVectorTarget"
    }
    if ((Get-Sha256Hex -Path $speakerVectorTarget) -ne $speakerVectorSha256) {
        throw "Deployed V2Pro speaker-vector model SHA-256 mismatch"
    }
    Write-Output "V2PRO_SPEAKER_VECTOR_MODEL_READY=$speakerVectorTarget"
}

if ($InstallFastLangDetect) {
    # fastText 0.9.2 language-identification model required by the vendored
    # GPT_SoVITS/text/LangSegmenter import (its inference cache directory).
    $fastLangDetectUrl = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
    $fastLangDetectSha256 = "7e69ec5451bc261cc7844e49e4792a85d7f09c06789ec800fc4a44aec362764e"
    $fastLangDetectName = "lid.176.bin"
    $fastLangDetectTarget = Join-Path $modelsRoot "gpt-sovits\fast_langdetect\$fastLangDetectName"

    if ($null -ne $FastLangDetectPath -and -not [string]::IsNullOrWhiteSpace($FastLangDetectPath)) {
        if ([System.IO.Path]::IsPathRooted($FastLangDetectPath)) {
            $fastLangDetectSource = [System.IO.Path]::GetFullPath($FastLangDetectPath)
        }
        else {
            $fastLangDetectSource = [System.IO.Path]::GetFullPath((Join-Path $root $FastLangDetectPath))
        }
        if (-not $fastLangDetectSource.StartsWith($modelsRootPrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
            throw "fast-langdetect model must remain under models/: $fastLangDetectSource"
        }
    }
    else {
        $downloadsDir = Join-Path $modelsRoot "downloads"
        New-Item -ItemType Directory -Force -Path $downloadsDir | Out-Null
        $fastLangDetectSource = Join-Path $downloadsDir $fastLangDetectName
        if (-not (Test-Path -LiteralPath $fastLangDetectSource -PathType Leaf)) {
            & curl.exe --ssl-no-revoke --fail --location --retry 3 --retry-all-errors --continue-at - --output $fastLangDetectSource $fastLangDetectUrl
            if ($LASTEXITCODE -ne 0) {
                throw "fast-langdetect model download failed with exit code $LASTEXITCODE"
            }
        }
    }

    if (-not (Test-Path -LiteralPath $fastLangDetectSource -PathType Leaf)) {
        throw "fast-langdetect model does not exist: $fastLangDetectSource"
    }
    if ((Get-Sha256Hex -Path $fastLangDetectSource) -ne $fastLangDetectSha256) {
        throw "fast-langdetect model SHA-256 mismatch"
    }

    if (-not (Test-Path -LiteralPath $fastLangDetectTarget -PathType Leaf)) {
        $targetParent = Split-Path -Parent $fastLangDetectTarget
        New-Item -ItemType Directory -Force -Path $targetParent | Out-Null
        if ([System.IO.Path]::GetFullPath($fastLangDetectSource) -ne [System.IO.Path]::GetFullPath($fastLangDetectTarget)) {
            Copy-Item -LiteralPath $fastLangDetectSource -Destination $fastLangDetectTarget
        }
    }
    if (-not (Test-Path -LiteralPath $fastLangDetectTarget -PathType Leaf)) {
        throw "fast-langdetect model was not deployed: $fastLangDetectTarget"
    }
    if ((Get-Sha256Hex -Path $fastLangDetectTarget) -ne $fastLangDetectSha256) {
        throw "Deployed fast-langdetect model SHA-256 mismatch"
    }
    Write-Output "FAST_LANGDETECT_MODEL_READY=$fastLangDetectTarget"
}

$verified = 0
$seenPaths = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::OrdinalIgnoreCase)
foreach ($line in Get-Content -LiteralPath $checksumPath) {
    if ([string]::IsNullOrWhiteSpace($line) -or $line.TrimStart().StartsWith("#")) {
        continue
    }
    $parts = $line -split "\s+", 2
    if ($parts.Count -ne 2) {
        throw "Invalid checksum line: $line"
    }
    $expected = $parts[0].ToLowerInvariant()
    if ($expected -notmatch "^[0-9a-f]{64}$") {
        throw "Invalid SHA-256: $($parts[0])"
    }
    $relativePath = $parts[1].TrimStart("*", " ")
    if ([System.IO.Path]::IsPathRooted($relativePath) -or $relativePath -match "(^|[\\/])\.\.([\\/]|$)") {
        throw "Model path escapes models root: $relativePath"
    }
    $modelPath = [System.IO.Path]::GetFullPath((Join-Path $root $relativePath))
    if (-not $modelPath.StartsWith($modelsRootPrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Model path escapes models root: $relativePath"
    }
    if (-not $seenPaths.Add($relativePath)) {
        throw "Duplicate model path: $relativePath"
    }
    if (-not (Test-Path -LiteralPath $modelPath -PathType Leaf)) {
        throw "Missing allowlisted model file: $relativePath"
    }
    $actual = Get-Sha256Hex -Path $modelPath
    if ($actual -ne $expected) {
        throw "SHA-256 mismatch for $relativePath"
    }
    $verified++
}

if ($verified -eq 0) {
    throw "model checksum manifest has no entries"
}

Write-Output "MODEL_CHECKSUMS_VERIFIED=$verified"

# Zero-shot inference loads its side models from the pinned checkout's own
# tree (GPT_SoVITS/sv.py hardcodes sv_path and the text frontend hardcodes
# the G2PW / fast-langdetect locations relative to the vendor working
# directory).  Link the checksum-verified copies under models/ into that tree
# so the vendored hardcoded paths resolve without duplicating the weights.
# This runs after manifest verification, so every link target is verified.
$vendorPackage = Join-Path $root "vendor\GPT-SoVITS\GPT_SoVITS"
if (-not (Test-Path -LiteralPath $vendorPackage -PathType Container)) {
    throw "Pinned GPT-SoVITS checkout is missing: $vendorPackage"
}
$vendorModelLinks = [ordered]@{
    # Keys are relative to the GPT_SoVITS package directory.
    "pretrained_models\chinese-hubert-base"           = "models\gpt-sovits\chinese-hubert-base"
    "pretrained_models\chinese-roberta-wwm-ext-large" = "models\gpt-sovits\bert"
    "pretrained_models\sv"                            = "models\gpt-sovits\sv"
    "pretrained_models\fast_langdetect"               = "models\gpt-sovits\fast_langdetect"
    "text\G2PWModel"                                  = "models\gpt-sovits\GPT_SoVITS\text\G2PWModel"
}
foreach ($link in $vendorModelLinks.GetEnumerator()) {
    $linkPath = Join-Path $vendorPackage $link.Key
    $linkTarget = Join-Path $root $link.Value
    if (-not (Test-Path -LiteralPath $linkTarget -PathType Container)) {
        throw "Vendor model link target is missing: $linkTarget"
    }
    if (Test-Path -LiteralPath $linkPath -PathType Container) {
        $existing = Get-Item -LiteralPath $linkPath -Force
        if ($existing.LinkType -ne "Junction") {
            throw "Vendor model path exists and is not a junction: $linkPath"
        }
    }
    else {
        $linkParent = Split-Path -Parent $linkPath
        New-Item -ItemType Directory -Force -Path $linkParent | Out-Null
        New-Item -ItemType Junction -Path $linkPath -Target $linkTarget | Out-Null
    }
}
Write-Output "VENDOR_PRETRAINED_LINKS=$($vendorModelLinks.Count)"
