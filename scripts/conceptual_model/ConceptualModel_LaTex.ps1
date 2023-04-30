Param(
    [string[]]$chapters
)

$texFile = "..\generated\conceptual_model\ConceptualModel.tex"
$texBodyFile = "..\generated\conceptual_model\ConceptualModel_body.tex"
$mdFile = "..\paper\conceptual_model\ConceptualModel.md"

if (!(Test-Path -Path ../generated/conceptual_model -PathType Container)){
    mkdir ../generated/conceptual_model
}
if ( Test-Path -Path $texFile -PathType Leaf )
{
    Remove-ItemSafely $texFile
}
if ( Test-Path -Path $texBodyFile -PathType Leaf )
{
    Remove-ItemSafely $texBodyFile
}

pandoc -s $mdFile $chapters -o $texFile --top-level-division=chapter
pandoc $mdFile $chapters -o $texBodyFile --top-level-division=chapter
if ( Test-Path -Path $texFile -PathType Leaf)
{
    if ( Test-Path -Path $texBodyFile -PathType Leaf)
    {
        Write-Output "`t- 2/2 LaTex Dokumente generiert"
    }
    else {
        Write-Output "`t- 1/2 LaTex Dokumente generiert"
    }
}