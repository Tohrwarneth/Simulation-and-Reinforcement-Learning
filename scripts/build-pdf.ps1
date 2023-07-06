<#
    .Description
    Generiert eine PDF aus einer Markdown Datei.

    Benoetigt: Latex (Empfohlen: MikTex), Pandoc mit dem abstract-section lua-Filter, bibtex, und pdflatex

    .PARAMETER status
    Schreibt den Status in grosser roter Schrift ueber den Titel.

    .PARAMETER outputDir
    Speicherort der PDF.

    .PARAMETER mainFile
    Haupt-Markdown-Datei, die die Meta-Daten für die PDF enthaelt und als erste Seite dient.

    .PARAMETER fileName
    Dateiname der PDF ohne Endung.

    .PARAMETER imageDir
    Speicherort der genutzten Bilder.

    .PARAMETER hsTemplate
    Soll die LaTex-Vorlage der Hochschule Trier verwendet werden?

    .PARAMETER saveLatex
    Soll die LaTex-Datei erhalten bleiben?

    .PARAMETER bibFile
    Speicherort der BibTex (*.bib).

    .PARAMETER secondarysRootDir
    Legt ein gemeinsamer Ordner für weitere Markdowns fest und wird vor deren Pfade eingefuegt. Wenn nicht gesetzt, wird nichts eingefuegt.

    .PARAMETER secondarys
    Weitere Markdowns, die in der vorkommenden Reihenfolge zusammengefuehrt werden.
#>

param(
    [Parameter(HelpMessage = "Schreibt den Status in grosser roter Schrift über den Titel.")]
    [Alias("ST")]
    [String]
    $status,
    [Parameter(Position = 0, Mandatory = $true, HelpMessage = "Speicherort der PDF.")]
    [Alias("OD")]
    [String]
    $outputDir,
    [Parameter(Position = 1, Mandatory = $true, HelpMessage = "Haupt-Markdown-Datei, die die Meta-Daten für die PDF enthaelt und als erste Seite dient.")]
    [Alias("MF")]
    [String]
    $mainFile,
    [Parameter(HelpMessage = "Dateiname der PDF ohne Endung.")]
    [Alias("FN")]
    [String]
    $fileName,
    [Parameter(HelpMessage = "Speicherort der genutzten Bilder.")]
    [Alias("ID")]
    [String]
    $imageDir,
    [Parameter(HelpMessage = "Soll die LaTex-Vorlage der Hochschule Trier verwendet werden?")]
    [Alias("HS")]
    [Switch]
    $hsTemplate,
    [Parameter(HelpMessage = "Hat das Dokument einen Abstract?")]
    [Alias("AB")]
    [Switch]
    $hasAbstract,
    [Parameter(HelpMessage = "Soll die LaTex-Datei erhalten bleiben?")]
    [Alias("SL")]
    [Switch]
    $saveLatex,
    [Parameter(HelpMessage = "Speicherort der BibTex (*.bib).")]
    [Alias("BF")]
    [String]
    $bibFile,
    [Parameter(HelpMessage = "Legt ein gemeinsamer Ordner für weitere Markdowns fest und wird vor deren Pfade eingefuegt. Wenn nicht gesetzt, wird nichts eingefuegt.")]
    [Alias("SR")]
    [String]
    $secondarysRootDir,
    [Parameter(Position = 2, ValueFromRemainingArguments, HelpMessage = "Weitere Markdowns, die in der vorkommenden Reihenfolge zusammengefuehrt werden.")]
    [String[]]
    $secondarys
)

Write-Output "Build..."
$warning = 0

if ($status)
{
    $fileName = "$( $fileName )_$( $status )"
}
else
{
    $status = ""
}

$pdfFile = Join-Path -Path $outputDir -ChildPath  "$( $fileName ).pdf"
$dir = $PSScriptRoot

if (!$secondarysRootDir)
{
    $secondarysRootDir = ""
}

$chaptersList = [String[]]@()
foreach ($c in $secondarys)
{
    $c = Join-Path -Path $secondarysRootDir -ChildPath  $c
    $chaptersList += $c
}

$imageDirAbsolut = (get-item $imageDir).FullName
Set-Location -Path (get-item $mainFile).Directory.FullName
$imageRelPath = Resolve-Path -Relative -Path $imageDirAbsolut
Set-Location -Path $dir

$tmpDir = Join-Path -Path $outputDir -ChildPath  "tmp"

$imgTmpDir = Join-Path -Path $outputDir -ChildPath  $imageRelPath
$latexFile = Join-Path -Path $tmpDir -ChildPath  "$( $fileName ).tex"

if (!(Test-Path -Path $tmpDir -PathType Container))
{
    New-Item -Path $outputDir -Name "tmp" -ItemType Directory | Out-Null
}

if ($bibFile)
{
    if ($hasAbstract)
    {
        pandoc  -s $mainFile $chaptersList --filter pandoc-crossref -o $latexFile --top-level-division=chapter `
        -f markdown -t latex `
        --template "./template/vorlage-project.tex" `
        --resource-path $tmpDir `
        --citeproc --bibliography=$bibFile `
        --lua-filter="filter/abstract-section.lua" `
        --metadata status=$status
    }
    else
    {
        pandoc  -s $mainFile $chaptersList --filter pandoc-crossref -o $latexFile --top-level-division=chapter `
        -f markdown -t latex `
        --template "./template/vorlage-project.tex" `
        --resource-path $tmpDir `
        --citeproc --bibliography=$bibFile `
        --metadata status=$status
    }
}
else
{
    if ($hasAbstract)
    {
        pandoc  -s $mainFile $chaptersList --filter pandoc-crossref -o $latexFile --top-level-division=chapter `
        -f markdown -t latex `
        --template "./template/vorlage-project.tex" `
        --resource-path $tmpDir `
        --lua-filter="filter/abstract-section.lua" `
        --metadata status=$status

    }
    else
    {
        pandoc  -s $mainFile $chaptersList --filter pandoc-crossref -o $latexFile --top-level-division=chapter `
        -f markdown -t latex `
        --template "./template/vorlage-project.tex" `
        --resource-path $tmpDir `
        --metadata status=$status
    }
}

if (!(Test-Path -Path $imgTmpDir -PathType Container))
{
    mkdir $imgTmpDir | Out-Null
}

if (Test-Path -Path $latexFile -PathType Leaf)
{
    Write-Output "`t- LaTex Dokumente generiert"
}
else
{
    Write-Warning "`t- Kein LaTex Dokumente generiert"
    exit
    $warning = 1
}

Copy-Item -Path "$( $imageDir )/*" -Destination $imgTmpDir -Recurse
Copy-Item -Path "./template/i-studis.cls" -Destination $tmpDir -Recurse
$tmpLogo = Join-Path -Path $tmpDir -ChildPath "images"
if (-Not (Test-Path -Path $tmpLogo -PathType Container))
{
    New-Item -Path $tmpDir -Name "images" -ItemType Directory | Out-Null
}
Copy-Item -Path "./template/logo-fb-informatik.pdf" -Destination $tmpLogo -Recurse

$latexFile = (get-item $latexFile).FullName
Set-Location -Path $tmpDir
Write-Output  (Get-Date) | Out-File -FilePath "Console.log"

if ($bibFile)
{
    $steps = 3
}
else
{
    $steps = 2
}

Write-Host -NoNewLine "`t- Render PDF: (1/$( $steps )) Steps"
pdflatex $latexFile | Out-File -append -FilePath "Console.log"
Write-Host -NoNewLine "`r`t- Render PDF: (2/$( $steps )) Steps"
bibtex $name | Out-Null
pdflatex $latexFile | Out-Null

if ($bibFile)
{
    Write-Host -NoNewLine "`r`t- Render PDF: (3/$( $steps )) Steps"
    pdflatex $latexFile.tex | Out-Null
}

if (Test-Path -Path "$( $fileName ).pdf" -PathType Leaf)
{
    $pdfFile = Move-Item -Force -PassThru -Path "$( $fileName ).pdf" -Destination ".."
    Write-Output "`r`t- PDF aus LaTex Dokument generiert: $( Resolve-Path -Path $pdfFile )"
    if ($saveLatex)
    {
        Move-Item -Force -PassThru -Path "$( $fileName ).tex" -Destination ".."
    }
}
else
{
    Write-Warning "`t- kein PDF aus LaTex Dokument generiert"
    $warning = 1
}

Set-Location -Path $dir

if (Test-Path $imgTmpDir -PathType Container)
{
    Remove-Item $imgTmpDir -Force -Recurse
}

if ((Test-Path $tmpDir -PathType Container ) -AND ($warning -eq 0))
{
    Remove-Item $tmpDir -Force -Recurse
}

if ($warning)
{
    Write-Warning "Fertig mit Warnungen"
}
else
{
    Write-Output "Fertig"
}