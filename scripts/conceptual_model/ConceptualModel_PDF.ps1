Param(
    [string[]]$name,
    [string[]]$chapters,
    [string[]]$inputPath,
    [string[]]$outputPath
)
function pdfGen
{


    $pdfFile = "$outputPath\$name.pdf"
    $mdFile = "$inputPath\$name.md"

    if (!(Test-Path -Path $outputPath -PathType Container))
    {
        mkdir $outputPath
    }
    if (Test-Path -Path $pdfFile -PathType Leaf)
    {
        Remove-ItemSafely $pdfFile
    }

    pandoc -s $mdFile $chapters -o $pdfFile
    #--template $template
    if (Test-Path -Path $pdfFile -PathType Leaf)
    {
        Write-Output "`t`t- PDF generiert"
    }
}
