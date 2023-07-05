param(
    [Parameter(HelpMessage = "Schreibt den Status in grosser roter Schrift über den Titel.")]
    [Alias("ST", "Status")]
    [String]
    $status,
    [Parameter(Mandatory = $true, HelpMessage = "Speicherort der PDF.")]
    [Alias("OD", "OutputDir")]
    [System.IO.FileInfo]
    [ValidateScript({
        if (-Not($_ | Test-Path))
        {
            throw "Ordner existiert nicht."
        }
        if (-Not($_ | Test-Path -PathType Container))
        {
            throw "OutputDir muss ein Ordner sein"
        }
        return $true
    })]
    $outputDir,
    [Parameter(Mandatory = $true, HelpMessage = "Haupt-Markdown-Datei, die die Meta-Daten für die PDF enthaelt und als erste Seite dient.")]
    [Alias("MF", "MainFile")]
    [System.IO.FileInfo]
    [ValidateScript({
        if (-Not($_ | Test-Path))
        {
            throw "Datei existiert nicht."
        }
        if (-Not($_ | Test-Path -PathType Leaf))
        {
            throw "MainFile muss eine Markdown Datei sein."
        }
        if ($_ -notmatch "(\.md)")
        {
            throw "MainFile muss ein Markdown (*.md) sein."
        }
        return $true
    })]
    $mainFile,
    [String(HelpMessage = "Dateiname der PDF ohne Endung.")]
    [Alias("FN", "FileName")]
    $fileName,
    [System.IO.FileInfo(HelpMessage = "Speicherort der genutzten Bilder.")]
    [Alias("ID", "ImageDir")]
    [ValidateScript({
        if (-Not($_ | Test-Path))
        {
            throw "Ordner existiert nicht."
        }
        if (-Not($_ | Test-Path -PathType Container))
        {
            throw "ImageDir muss ein Ordner sein"
        }
        return $true
    })]
    $imageDir,
    [Switch(HelpMessage = "Soll die LaTex-Vorlage der Hochschule Trier verwendet werden?")]
    [Alias("HS", "HSTemplate")]
    $hsTemplate,
    [Switch(HelpMessage = "Soll die LaTex-Datei erhalten bleiben?")]
    [Alias("SL", "SaveLatex")]
    $saveLatex,
    [System.IO.FileInfo(HelpMessage = "Speicherort der BibTex (*.bib).")]
    [Alias("BF", "BibFile")]
    [ValidateScript({
        if (-Not($_ | Test-Path))
        {
            throw "Datei existiert nicht."
        }
        if (-Not($_ | Test-Path -PathType Leaf))
        {
            throw "BibFile muss eine Datei sein"
        }
        if ($_ -notmatch "(\.bib)")
        {
            throw "BibFile muss eine BibTex-Datei (*.bib) sein."
        }
        return $true
    })]
    $bibFile,
    [System.IO.FileInfo(HelpMessage = "Legt ein gemeinsamer Ordner für weitere Markdowns fest und wird vor deren Pfade eingefuegt. Wenn nicht gesetzt, wird nichts eingefuegt.")]
    [Alias("SR", "SecRoot")]
    [ValidateScript({
        if (-Not($_ | Test-Path))
        {
            throw "Ordner existiert nicht."
        }
        if (-Not($_ | Test-Path -PathType Container))
        {
            throw "SecRoot muss ein Ordner sein"
        }
        return $true
    })]
    $secondarysRootDir,
    [System.IO.FileInfo[](ValueFromRemainingArguments, HelpMessage = "Weitere Markdowns, die in der vorkommenden Reihenfolge zusammengefuehrt werden.")]
    [ValidateScript({
        if (-Not($_ | Test-Path))
        {
            throw "Datei existiert nicht."
        }
        if (-Not($_ | Test-Path -PathType Leaf))
        {
            throw "Die Weitere Markdowns müssen Dateien sein."
        }
        return $true
    })]
    $secondarys
)
Write-Output "Build..."
$warning = 0

if ($status)
{
    $fileName = "$( $fileName )_$( $status )"
}

$pdfFile = Join-Path -Path $outputDir -ChildPath  "$( $fileName ).pdf"
$latexFile = Join-Path -Path $outputDir -ChildPath  "$( $fileName ).tex"

if (!$secondarysRootDir)
{
    $secondarysRootDir = ""
}

$chaptersList = [System.Collections.ArrayList]@()
foreach ($c in $secondarys)
{
    $c = Join-Path -Path $secondarysRootDir -ChildPath  $c
    $chaptersList.Add($c)
}

$imageRelPath = Resolve-RelativePath -Path $imagePath -FromDirectory $mainFile

$tmpPath = Join-Path -Path $outputDir -ChildPath  "tmp"

$imgTmpPath = Join-Path -Path $outputPath -ChildPath  $imageRelPath

if (!(Test-Path -Path $tmpPath -PathType Container))
{
    New-Item -Path $outputPath -Name "tmp" -ItemType Directory | Out-Null
}

pandoc -s $mainFile $chaptersList --filter pandoc-crossref -o $texFile --top-level-division=chapter `
--template "./template/vorlage-project.tex" `
--resource-path $tmpPath `
--citeproc --bibliography = $bibFile `
--lua-filter = "filter/abstract-section.lua" `
--metadata status = $status

if (!(Test-Path -Path $imgTmpPath -PathType Container))
{
    mkdir $imgTmpPath | Out-Null
}

if (Test-Path -Path $texFile -PathType Leaf)
{
    Write-Output "`t- LaTex Dokumente generiert"
}
else
{
    Write-Warning "`t- Kein LaTex Dokumente generiert"
    $warning = 1
}

Copy-Item -Path "$( $imagePath )/*" -Destination $imgTmpPath -Recurse
Copy-Item -Path "./template/i-studis.cls" -Destination $tmpPath
Copy-Item -Path "./template/logo-fb-informatik.pdf" -Destination $imgTmpPath

$dir = $PSScriptRoot
#$pdfFile = "$( $name ).pdf"
#$pdfPath = Join-Path $tmpPath -childPath $pdfFile

#if (Test-Path $pdfFile)
#{
#    Remove-Item $pdfFile
#}

Set-Location -Path $tmpPath
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

if (Test-Path -Path $pdfFile -PathType Leaf)
{
    $pdfFile = Move-Item -Force -PassThru -Path "$( $fileName ).pdf" -Destination ".."
    Write-Output "`r`t- PDF aus LaTex Dokument generiert: $( Resolve-Path -Path $pdfFile )"
}
else
{
    Write-Warning "`t- kein PDF aus LaTex Dokument generiert"
    $warning = 1
}

Set-Location -Path $dir
Set-Location -Path ../../

if (Test-Path $imgTmpPath)
{
    Remove-Item $imgTmpPath
}

if (Test-Path $tmpPath AND $warning -eq 0)
{
    Remove-Item $tmpPath
}

if ($warning)
{
    Write-Warning "Fertig mit Warnungen"
}
else
{
    Write-Output "Fertig"
}