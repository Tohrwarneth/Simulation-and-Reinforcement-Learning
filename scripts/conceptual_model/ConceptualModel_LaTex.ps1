Param(
    [string[]]$name,
    [string[]]$chapters,
    [string[]]$images,
    [string[]]$inputPath,
    [string[]]$outputPath
)

$template = "../paper/template/tex/vorlage-project.tex"


$texFile = "$name.tex"
$texFile = Join-Path -Path $outputPath -ChildPath $texFile
$texBodyFile = "$($name)_body.tex"
$texBodyFile = Join-Path -Path $outputPath -ChildPath $texBodyFile
$mdFile = "$name.md"
$mdFile = Join-Path -Path $inputPath -ChildPath  $mdFile
$tmpPath = Join-Path -Path $outputPath -ChildPath  "tmp"
$imgTmpPath = Join-Path -Path $tmpPath -ChildPath  "images"
$imgOutputPath = Join-Path -Path $outputPath -ChildPath  "images"
$imgInputPath = Join-Path -Path $inputPath -ChildPath  "images"

if (!(Test-Path -Path $outputPath -PathType Container))
{
    mkdir $outputPath | Out-Null
}
if (Test-Path -Path $texFile -PathType Leaf)
{
    Remove-ItemSafely $texFile
}
if (Test-Path -Path $texBodyFile -PathType Leaf)
{
    Remove-ItemSafely $texBodyFile
}

pandoc -s $mdFile $chapters -o $texFile --top-level-division=chapter --template $template
pandoc $mdFile $chapters -o $texBodyFile --top-level-division=chapter --template $template



if (!(Test-Path -Path $imgTmpPath -PathType Leaf))
{
    mkdir $imgTmpPath | Out-Null
}

if (!(Test-Path -Path $imgOutputPath -PathType Leaf))
{
    mkdir $imgOutputPath | Out-Null
}

if (Test-Path -Path $texFile -PathType Leaf)
{
    if (Test-Path -Path $texBodyFile -PathType Leaf)
    {
        Write-Output "`t`t- 2/2 LaTex Dokumente generiert"
    }
    else
    {
        Write-Output "`t`t- 1/2 LaTex Dokumente generiert"
    }
}
else
{
    Write-Output "`t`t- 0/2 LaTex Dokumente generiert"
}

for ($i = 0; $i -lt $images.Count; $i++) {
    Copy-Item -Path "$($images.Get($i))/*" -Destination $imgOutputPath -Recurse
}
Copy-Item -Path "../paper/template/tex/i-studis.cls" -Destination $tmpPath
Copy-Item -Path "../paper/template/tex/logo-fb-informatik.pdf" -Destination $imgTmpPath

$dir = $PSScriptRoot

Set-Location -Path $tmpPath
pdflatex "../$name.tex" | Out-File -FilePath "Console.log"
pdflatex "../$name.tex" | Out-Null

if (Test-Path -Path "$name.pdf" -PathType Leaf)
{
    Copy-Item -Path "$name.pdf" -Destination "../$($name)-LaTex.pdf"
    Write-Output "`t`t- PDF aus LaTex Dokument generiert"
    Write-Output "../$($name)-LaTex.pdf"
}
else
{
    Write-Output "`t`t- kein PDF aus LaTex Dokument generiert"
}

Set-Location -Path $dir
Set-Location -Path ../