Param(
    [string[]]$chapters,
    [string]$template
)

$texFile = "..\generated\conceptual_model\ConceptualModel.tex"
$texBodyFile = "..\generated\conceptual_model\ConceptualModel_body.tex"
$mdFile = "..\paper\conceptual_model\ConceptualModel.md"

if (!(Test-Path -Path ../generated/conceptual_model -PathType Container))
{
    mkdir ../generated/conceptual_model | Out-Null
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



if (!(Test-Path -Path "../generated/conceptual_model/tmp/images" -PathType Leaf))
{
    mkdir "../generated/conceptual_model/tmp/images" | Out-Null
}

if (!(Test-Path -Path "../generated/conceptual_model/images" -PathType Leaf))
{
    mkdir "../generated/conceptual_model/images" | Out-Null
}

if (Test-Path -Path $texFile -PathType Leaf)
{
    if (Test-Path -Path $texBodyFile -PathType Leaf)
    {
        Write-Output "`t- 2/2 LaTex Dokumente generiert"
    }
    else
    {
        Write-Output "`t- 1/2 LaTex Dokumente generiert"
    }
}
else
{
    Write-Output "`t- 0/2 LaTex Dokumente generiert"
}

Copy-Item -Path "../paper/conceptual_model/images/*" -Destination "../generated/conceptual_model/images" -Recurse
Copy-Item -Path "../paper/template/tex/i-studis.cls" -Destination "../generated/conceptual_model/tmp"
Copy-Item -Path "../paper/template/tex/logo-fb-informatik.pdf" -Destination "../generated/conceptual_model/tmp/images"

$dir = $PSScriptRoot

Set-Location -Path ../generated/conceptual_model/tmp
pdflatex ../ConceptualModel.tex | Out-File -FilePath "Console.log"
pdflatex ../ConceptualModel.tex | Out-Null

if (Test-Path -Path ConceptualModel.pdf -PathType Leaf)
{
    Copy-Item -Path ConceptualModel.pdf -Destination ../ConceptualModel-LaTex.pdf
    Write-Output "`t- PDF aus LaTex Dokument generiert"
}
else
{
    Write-Output "`t- kein PDF aus LaTex Dokument generiert"
}

Set-Location -Path $dir
Set-Location -Path ../