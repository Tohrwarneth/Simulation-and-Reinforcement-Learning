$n = 3
$i = 0

function PrintGeneralProgress
{
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

function PrintInnerProgress
{
    param([int]$id = 1, [int] $step = 0, [int] $max = 1, [string] $activity = "Build", [string] $operation = "")
    $j++
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
$chapters = $mdFileIdentity, $mdFileProperties, $mdFileBehavior, $mdFileVerification
$images = "..\paper\conceptual_model\images", "..\paper\simulation\images"
$m = (Get-ChildItem 'conceptual_model' | Measure-Object).Count
$j = 0
Write-Output "`tBuild Conceptual Model"
Get-ChildItem 'conceptual_model' | ForEach-Object {
    & $_.FullName -chapters $chapters -inputPath "..\paper\conceptual_model\" -outputPath "..\generated\conceptual_model\" -name "ConceptualModel" -images $images
    PrintInnerProgress -step $j -max $m -activity "Conceptual Model" -operation "Build Conecptual Model"
}

PrintGeneralProgress

# Simulation

$mdFileSimOverview = "..\paper\simulation\chapters\overview.md"
$mdFileSimPerson = "..\paper\simulation\chapters\person.md"
$mdFileSimElevator = "..\paper\simulation\chapters\elevator.md"
$chapters = $mdFileIdentity, $mdFileProperties, $mdFileBehavior, $mdFileVerification, `
            $mdFileSimOverview, $mdFileSimPerson, $mdFileSimElevator
$m = (Get-ChildItem 'conceptual_model' | Measure-Object).Count
$j = 0
Write-Output "`tBuild Simulation"
Get-ChildItem 'conceptual_model' | ForEach-Object {
    & $_.FullName -chapters $chapters -inputPath "..\paper\simulation\" -outputPath "..\generated\simulation\" -name "Simulation" -images $images
    PrintInnerProgress -step $j -max $m -activity "Simulation" -operation "Build Simulation Paper"
}

PrintGeneralProgress

# Reinforcement
$images = "..\paper\reinforcement\images"
$mdFileSimOverview = "..\paper\reinforcement\chapters\overview.md"
$mdFileSimKonzept = "..\paper\reinforcement\chapters\konzept.md"
$chapters = $mdFileSimOverview, $mdFileSimKonzept
$m = (Get-ChildItem 'conceptual_model' | Measure-Object).Count
$j = 0
Write-Output "`tBuild Reinforcement"
Get-ChildItem 'conceptual_model' | ForEach-Object {
    & $_.FullName -chapters $chapters -inputPath "..\paper\reinforcement\" -outputPath "..\generated\reinforcement\" -name "Reinforcement" -images $images
    PrintInnerProgress -step $j -max $m -activity "Reinforcement" -operation "Build Reinforcement Paper"
}

PrintGeneralProgress

Write-Host "Build Fertig"