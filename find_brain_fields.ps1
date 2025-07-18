$patterns = @('\.brain_entities', '\.brain_relationships')
$results = @()

Get-ChildItem -Path @('src\cognitive\*.rs', 'src\core\*.rs', 'tests\*.rs') -File | ForEach-Object {
    $file = $_
    $content = Get-Content $file.FullName
    for ($i = 0; $i -lt $content.Count; $i++) {
        $line = $content[$i]
        foreach ($pattern in $patterns) {
            if ($line -match $pattern) {
                $results += [PSCustomObject]@{
                    File = $file.FullName.Replace((Get-Location).Path + '\', '')
                    LineNumber = $i + 1
                    Line = $line.Trim()
                    Pattern = $pattern
                }
            }
        }
    }
}

$results | Sort-Object File, LineNumber | Format-Table -AutoSize