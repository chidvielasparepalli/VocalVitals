$ErrorActionPreference = "Stop"
$ModelUrl = "https://vercel-demo-gqf7mx8p9-chidvigamer-5480s-projects.vercel.app/swallow_binary.onnx"
$OutFile = Join-Path $PSScriptRoot "swallow_binary.onnx"
Invoke-WebRequest -Uri $ModelUrl -OutFile $OutFile
Write-Host "Model downloaded to $OutFile"
