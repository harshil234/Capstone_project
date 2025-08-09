@echo off
echo ========================================
echo AdVision AI Presentation Setup
echo ========================================
echo.

echo Creating presentation package directory...
if not exist "AdVision_Presentation" mkdir "AdVision_Presentation"

echo Copying files to presentation directory...
copy "AdVision_Presentation_VBA.bas" "AdVision_Presentation\"
copy "PowerPoint_Presentation_Guide.md" "AdVision_Presentation\"
copy "Presentation_Slide_Content.md" "AdVision_Presentation\"
copy "README.md" "AdVision_Presentation\"

echo.
echo ========================================
echo Setup Complete!
echo ========================================
echo.
echo Files have been organized in the 'AdVision_Presentation' folder.
echo.
echo Next steps:
echo 1. Open Microsoft PowerPoint
echo 2. Press Alt+F11 to open VBA editor
echo 3. Import the VBA code from AdVision_Presentation_VBA.bas
echo 4. Follow the setup guide in PowerPoint_Presentation_Guide.md
echo.
echo Press any key to open the presentation folder...
pause >nul

explorer "AdVision_Presentation" 