Param(
    [string[]]$chapters,
    [string]$template
)

$pdfFile = "..\generated\conceptual_model\ConceptualModel.pdf"
$mdFile = "..\paper\conceptual_model\ConceptualModel.md"

if (!(Test-Path -Path ../generated/conceptual_model -PathType Container)){
    mkdir ../generated/conceptual_model
}
if ( Test-Path -Path $pdfFile -PathType Leaf )
{
    Remove-ItemSafely $pdfFile
}

pandoc -s $mdFile $chapters -o $pdfFile
#--template $template
if ( Test-Path -Path $pdfFile -PathType Leaf)
{
    Write-Output "`t- PDF generiert"
}