# AdVision AI Presentation - Installation Guide

## 📋 Prerequisites

Before installing the AdVision AI presentation package, ensure you have:

### Required Software:
- **Microsoft PowerPoint** (2016, 2019, 2021, or Microsoft 365)
- **Windows 10/11** (for VBA functionality)
- **PowerShell** (for advanced setup scripts)

### Optional Software:
- **Visual Studio Code** or any text editor (for viewing markdown files)
- **Git** (for version control)

## 🚀 Installation Methods

### Method 1: Manual Installation (Recommended)

#### Step 1: Extract Package
1. Download the `AdVision_Presentation_Package.zip` file
2. Extract it to a folder on your computer
3. Navigate to the extracted folder

#### Step 2: Open PowerPoint
1. Launch Microsoft PowerPoint
2. Create a new presentation or open an existing one
3. Save the presentation as `.pptx` or `.pptm` format

#### Step 3: Import VBA Code
1. Press `Alt + F11` to open the VBA editor
2. In the Project Explorer (left panel), right-click on your presentation name
3. Select "Insert" → "Module"
4. Open the `AdVision_Presentation_VBA.bas` file in a text editor
5. Copy the entire content (Ctrl+A, Ctrl+C)
6. Paste it into the new module in PowerPoint (Ctrl+V)
7. Save the presentation

#### Step 4: Enable Macros
1. Go to File → Options → Trust Center
2. Click "Trust Center Settings"
3. Select "Macro Settings"
4. Choose "Enable all macros" (for development)
5. Click "OK"

#### Step 5: Initialize Presentation
1. In the VBA editor, press F5 or click the "Run" button
2. Select the `InitializePresentation` function
3. Click "Run"
4. You should see a success message

### Method 2: Automated Setup (Windows)

#### Step 1: Run Setup Script
1. Open PowerShell as Administrator
2. Navigate to the package folder
3. Run the setup script:
   ```powershell
   .\setup_presentation.ps1 -Validate -CreateTemplate
   ```

#### Step 2: Follow Generated Instructions
The script will:
- Validate all package files
- Create a basic PowerPoint template
- Show detailed setup instructions

### Method 3: Batch File Setup

#### Step 1: Run Batch File
1. Double-click `setup_presentation.bat`
2. The script will organize files and open the presentation folder

#### Step 2: Manual VBA Import
Follow the manual installation steps 2-5 above.

## 🔧 Configuration

### Customizing Colors
The VBA code uses a consistent color scheme. To customize:

1. Open the VBA editor (`Alt + F11`)
2. Find the color definitions in the code:
   ```vba
   RGB(37, 99, 235)  ' Primary Blue
   RGB(34, 197, 94)  ' Success Green
   RGB(245, 158, 11) ' Warning Orange
   RGB(168, 85, 247) ' Purple
   RGB(236, 72, 153) ' Pink
   ```
3. Replace with your preferred colors

### Updating Demo Data
To customize the demo data:

1. Find the `InitializeDemoData()` function in the VBA code
2. Modify the values in the `ActivePresentation.Tags.Add` calls
3. Update the demo functions with your own examples

### Adding Custom Functions
To add your own VBA functions:

1. Create a new module or add to the existing one
2. Follow the naming convention: `Sub FunctionName()`
3. Add error handling and comments
4. Test thoroughly before presenting

## 📋 Verification Checklist

After installation, verify the following:

### ✅ VBA Code
- [ ] VBA code imported successfully
- [ ] No compilation errors in VBA editor
- [ ] `InitializePresentation()` function runs without errors
- [ ] All demo functions are available

### ✅ PowerPoint Settings
- [ ] Macros enabled in Trust Center
- [ ] Presentation saved as `.pptx` or `.pptm`
- [ ] VBA editor accessible (`Alt + F11`)

### ✅ Demo Functions
- [ ] `RunDemo()` shows the demo menu
- [ ] `RunPerformanceDemo()` displays performance data
- [ ] `RunROIDemo()` shows ROI calculations
- [ ] `RunCopyGenerationDemo()` displays ad copy
- [ ] `RunImageGenerationDemo()` shows image generation
- [ ] `RunChatbotDemo()` displays chatbot interactions

### ✅ Utility Functions
- [ ] `ShowModelStatus()` displays model information
- [ ] `CreateProjectStructure()` shows file organization
- [ ] `OpenProjectWebsite()` opens browser (if app is running)
- [ ] `ExportToPDF()` creates PDF export

## 🛠️ Troubleshooting

### Common Issues and Solutions

#### Issue: "Macros are disabled"
**Solution:**
1. Go to File → Options → Trust Center → Trust Center Settings
2. Select "Macro Settings"
3. Choose "Enable all macros"
4. Click "OK"

#### Issue: "VBA editor not opening"
**Solution:**
1. Ensure PowerPoint is not in protected view
2. Check if VBA add-in is enabled
3. Try restarting PowerPoint

#### Issue: "Functions not working"
**Solution:**
1. Check for compilation errors in VBA editor
2. Ensure all code is properly pasted
3. Verify function names match exactly
4. Check for missing dependencies

#### Issue: "Demo buttons not appearing"
**Solution:**
1. Run `CreateNavigationButtons()` function
2. Check if slide layout supports shapes
3. Verify slide has title placeholder

#### Issue: "Website not opening"
**Solution:**
1. Ensure AdVision application is running on localhost:5000
2. Check firewall settings
3. Verify browser is set as default

### Debug Mode
To enable debug mode:

1. In VBA editor, add debug statements:
   ```vba
   Debug.Print "Function executed successfully"
   ```
2. Open Immediate Window (Ctrl+G)
3. Run functions and check output

### Error Logging
The VBA code includes error handling. To view errors:

1. Check the Immediate Window in VBA editor
2. Look for error messages in message boxes
3. Review the application log if available

## 📚 Additional Resources

### Documentation Files
- `README.md` - Package overview and quick start
- `PowerPoint_Presentation_Guide.md` - Detailed usage guide
- `Presentation_Slide_Content.md` - Slide content and structure
- `INSTALLATION_GUIDE.md` - This file

### Online Resources
- [Microsoft VBA Documentation](https://docs.microsoft.com/en-us/office/vba/)
- [PowerPoint VBA Reference](https://docs.microsoft.com/en-us/office/vba/api/powerpoint)
- [AdVision AI Project Documentation](http://localhost:5000)

### Support
For additional support:
1. Check the troubleshooting section above
2. Review the inline comments in the VBA code
3. Consult Microsoft's VBA documentation
4. Test functions in the VBA editor debug mode

## 🎯 Next Steps

After successful installation:

1. **Create Slides**: Follow the content guide in `Presentation_Slide_Content.md`
2. **Test Demos**: Run all demo functions to ensure they work
3. **Customize Content**: Update with your specific project data
4. **Practice Presentation**: Rehearse the flow and timing
5. **Prepare Backup**: Create screenshots and static versions
6. **Present**: Deliver your professional AdVision AI presentation!

## 📞 Contact

For technical support or questions:
- Review this installation guide
- Check the troubleshooting section
- Consult the VBA code comments
- Refer to Microsoft's VBA documentation

---

**Created for AdVision AI Project**  
**AI-Powered Advertising Analytics Platform**

*This installation guide ensures a smooth setup process for your interactive AdVision AI presentation.* 