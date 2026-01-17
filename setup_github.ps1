# Setup script to move EvlfEris to separate repo and push to GitHub
# 
# INSTRUCTIONS:
# 1. Create a new GitHub repo at: https://github.com/new
#    Name it: EvlfEris
#    Make it public or private (your choice)
#    DON'T add README, .gitignore, or license (we have them)
# 
# 2. Run this script with your GitHub repo URL:
#    .\setup_github.ps1 -GitHubUrl "https://github.com/YOUR_USERNAME/EvlfEris.git"

param(
    [Parameter(Mandatory=$true)]
    [string]$GitHubUrl
)

Write-Host "=================================" -ForegroundColor Cyan
Write-Host "  Evlf Eris - GitHub Setup" -ForegroundColor Cyan
Write-Host "=================================" -ForegroundColor Cyan
Write-Host ""

# Step 1: Move folder to separate location
Write-Host "📁 Step 1: Moving EvlfEris to separate directory..." -ForegroundColor Yellow

$currentPath = "c:\Users\wind xebec\Evlf\EvlfEris"
$newPath = "c:\Users\wind xebec\EvlfEris"

if (Test-Path $newPath) {
    Write-Host "❌ Error: $newPath already exists!" -ForegroundColor Red
    Write-Host "   Please remove or rename it first." -ForegroundColor Red
    exit 1
}

if (-not (Test-Path $currentPath)) {
    Write-Host "❌ Error: $currentPath not found!" -ForegroundColor Red
    exit 1
}

Move-Item -Path $currentPath -Destination $newPath
Write-Host "✅ Moved to: $newPath" -ForegroundColor Green
Write-Host ""

# Step 2: Initialize git (if not already done)
Write-Host "🔧 Step 2: Setting up Git..." -ForegroundColor Yellow
Set-Location $newPath

if (-not (Test-Path ".git")) {
    git init
    Write-Host "✅ Git initialized" -ForegroundColor Green
} else {
    Write-Host "✅ Git already initialized" -ForegroundColor Green
}
Write-Host ""

# Step 3: Add all files and commit
Write-Host "📝 Step 3: Committing files..." -ForegroundColor Yellow

git add .
git commit -m "Initial commit: Evlf Eris - Model Surgery and Pruning for Llama-3.2-3B"

Write-Host "✅ Files committed" -ForegroundColor Green
Write-Host ""

# Step 4: Add GitHub remote and push
Write-Host "🚀 Step 4: Pushing to GitHub..." -ForegroundColor Yellow

# Check if remote already exists
$existingRemote = git remote get-url origin 2>$null
if ($existingRemote) {
    Write-Host "⚠️  Remote 'origin' already exists: $existingRemote" -ForegroundColor Yellow
    $response = Read-Host "Replace it? (y/n)"
    if ($response -eq 'y') {
        git remote remove origin
        git remote add origin $GitHubUrl
    }
} else {
    git remote add origin $GitHubUrl
}

# Rename branch to main (if needed)
$currentBranch = git branch --show-current
if ($currentBranch -ne "main") {
    git branch -M main
}

# Push to GitHub
Write-Host "Pushing to: $GitHubUrl" -ForegroundColor Cyan
git push -u origin main

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "=================================" -ForegroundColor Green
    Write-Host "  ✅ SUCCESS!" -ForegroundColor Green
    Write-Host "=================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "Your repo is now at:" -ForegroundColor White
    Write-Host "  Local:  $newPath" -ForegroundColor Cyan
    Write-Host "  GitHub: $($GitHubUrl.Replace('.git', ''))" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Next steps:" -ForegroundColor Yellow
    Write-Host "1. Download base model: See SETUP.md" -ForegroundColor White
    Write-Host "2. Run model surgery: python surgery/prune.py" -ForegroundColor White
    Write-Host "3. Chat with Eris: .\run_eris.ps1" -ForegroundColor White
    Write-Host ""
} else {
    Write-Host ""
    Write-Host "❌ Push failed!" -ForegroundColor Red
    Write-Host "Make sure you:" -ForegroundColor Yellow
    Write-Host "  1. Created the GitHub repo" -ForegroundColor White
    Write-Host "  2. Have git credentials set up" -ForegroundColor White
    Write-Host "  3. Used the correct repo URL" -ForegroundColor White
    Write-Host ""
}
