# AdVision AI Presentation Setup Script
# PowerShell script for advanced setup and validation

param(
    [switch]$Validate,
    [switch]$CreateTemplate,
    [switch]$Help
)

function Show-Help {
    Write-Host @"
AdVision AI Presentation Setup Script

Usage:
    .\setup_presentation.ps1 [options]

Options:
    -Validate        Validate all files in the package
    -CreateTemplate  Create a basic PowerPoint template
    -Help           Show this help message

Examples:
    .\setup_presentation.ps1 -Validate
    .\setup_presentation.ps1 -CreateTemplate
    .\setup_presentation.ps1 -Validate -CreateTemplate
"@
}

function Test-FileExists {
    param([string]$FilePath)
    
    if (Test-Path $FilePath) {
        Write-Host "✅ $FilePath" -ForegroundColor Green
        return $true
    } else {
        Write-Host "❌ $FilePath" -ForegroundColor Red
        return $false
    }
}

function Validate-Package {
    Write-Host "Validating AdVision Presentation Package..." -ForegroundColor Cyan
    Write-Host "=============================================" -ForegroundColor Cyan
    
    $files = @(
        "AdVision_Presentation_VBA.bas",
        "PowerPoint_Presentation_Guide.md",
        "Presentation_Slide_Content.md",
        "README.md"
    )
    
    $allValid = $true
    
    foreach ($file in $files) {
        if (-not (Test-FileExists $file)) {
            $allValid = $false
        }
    }
    
    if ($allValid) {
        Write-Host "`n🎉 All files validated successfully!" -ForegroundColor Green
        Write-Host "Package is ready to use." -ForegroundColor Green
    } else {
        Write-Host "`n⚠️  Some files are missing. Please check the package." -ForegroundColor Yellow
    }
    
    return $allValid
}

function Create-PresentationTemplate {
    Write-Host "Creating PowerPoint Template..." -ForegroundColor Cyan
    Write-Host "=================================" -ForegroundColor Cyan
    
    try {
        # Check if PowerPoint is available
        $powerpoint = New-Object -ComObject PowerPoint.Application
        $powerpoint.Visible = $true
        
        # Create new presentation
        $presentation = $powerpoint.Presentations.Add()
        
        # Set up basic slide structure
        $slide1 = $presentation.Slides.Add(1, 1) # Title slide
        $slide1.Shapes.Title.TextFrame.TextRange.Text = "AdVision AI"
        $slide1.Shapes.Item(2).TextFrame.TextRange.Text = "AI-Powered Advertising Analytics Platform"
        
        # Add more slides based on content guide
        $slide2 = $presentation.Slides.Add(2, 2) # Content slide
        $slide2.Shapes.Title.TextFrame.TextRange.Text = "Project Overview"
        
        $slide3 = $presentation.Slides.Add(3, 2)
        $slide3.Shapes.Title.TextFrame.TextRange.Text = "Technical Architecture"
        
        $slide4 = $presentation.Slides.Add(4, 2)
        $slide4.Shapes.Title.TextFrame.TextRange.Text = "Core Features Demo"
        
        # Save the template
        $templatePath = "AdVision_Presentation_Template.pptx"
        $presentation.SaveAs((Get-Location).Path + "\" + $templatePath)
        
        Write-Host "✅ Template created: $templatePath" -ForegroundColor Green
        Write-Host "You can now import the VBA code into this template." -ForegroundColor Green
        
        # Close PowerPoint
        $presentation.Close()
        $powerpoint.Quit()
        
    } catch {
        Write-Host "❌ Error creating template: $($_.Exception.Message)" -ForegroundColor Red
        Write-Host "Please create the presentation manually and import the VBA code." -ForegroundColor Yellow
    }
}

function Show-SetupInstructions {
    Write-Host "`n📋 Setup Instructions:" -ForegroundColor Cyan
    Write-Host "=====================" -ForegroundColor Cyan
    Write-Host @"

1. Open Microsoft PowerPoint
2. Press Alt+F11 to open VBA editor
3. Right-click on your presentation name in Project Explorer
4. Select 'Insert' → 'Module'
5. Copy and paste the content from 'AdVision_Presentation_VBA.bas'
6. Save the presentation as .pptx or .pptm
7. Enable macros in Trust Center settings
8. Run InitializePresentation() function
9. Create slides following the content guide

For detailed instructions, see 'PowerPoint_Presentation_Guide.md'
"@
}

# Main script execution
if ($Help) {
    Show-Help
    exit
}

Write-Host "AdVision AI Presentation Setup Script" -ForegroundColor Magenta
Write-Host "=====================================" -ForegroundColor Magenta
Write-Host ""

if ($Validate) {
    Validate-Package
}

if ($CreateTemplate) {
    Create-PresentationTemplate
}

if (-not $Validate -and -not $CreateTemplate) {
    # Default behavior - validate and show instructions
    Validate-Package
    Show-SetupInstructions
}

Write-Host "`n🚀 Ready to create your AdVision AI presentation!" -ForegroundColor Green 