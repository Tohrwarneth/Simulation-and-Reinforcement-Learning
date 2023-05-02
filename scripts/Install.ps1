# winget install --id Microsoft.Powershell --source winget
# Set-ExecutionPolicy -ExecutionPolicy RemoteSigned

if (Get-Module -ListAvailable -Name Recycle) {
    Write-Host "Module exists"
}
else {
    Write-Host "Module does not exist"
    Install-Module -Name Recycle
}