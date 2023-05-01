$n = 2
$i = 0

function PrintGeneralProgress {
    $i++
    $OuterLoopProgressParameters = @{
        id = 0
        Activity = 'Build SRL Texte'
        Status = 'Progress:'
        PercentComplete = ($i/$n) * 100
        CurrentOperation = 'Build'
    }

    Write-Progress @OuterLoopProgressParameters
}

function PrintInnerProgress {
    param([int]$id=1, [int] $step=0, [int] $max=1, [string] $activity="Build", [string] $operation="")
    $InnerLoopProgressParameters = @{
        ParentId = 0
        ID = $id
        Activity = $activity
        Status = 'Progress:'
        PercentComplete = ($step/$max) * 100
        CurrentOperation = $operation
    }

    Write-Progress @InnerLoopProgressParameters
}

Write-Host "Build"

if (Test-Path -Path ../generated -PathType Container)
{
    Remove-ItemSafely ../generated
}


PrintGeneralProgress

# ConecptualModel

$mdFileIdentity = "..\paper\conceptual_model\chapters\identity.md"
$mdFileProperties = "..\paper\conceptual_model\chapters\properties.md"
$mdFileBehavior = "..\paper\conceptual_model\chapters\behavior.md"
$mdFileVerification = "..\paper\conceptual_model\chapters\verification.md"
$template = "../paper/template/tex/vorlage-project.tex"
$chapters = $mdFileIdentity, $mdFileProperties, $mdFileBehavior, $mdFileVerification
$m = (Get-ChildItem 'conceptual_model' | Measure-Object).Count
$j = 0
Get-ChildItem 'conceptual_model' | ForEach-Object {
    & $_.FullName -chapters $chapters -template $template
    $j = $j + 1
    PrintInnerProgress -step $j -max $m -activity "Conceptual Model" -operation "Build Conecptual Model"
}

PrintGeneralProgress

Write-Host "Build Fertig"