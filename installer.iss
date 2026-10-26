[Setup]
AppId={{329E0D17-54DD-493E-86F5-FAC513D32627}
AppName=BL62 Reconstruction
AppVersion=1.0.1
AppPublisher=BL62
AppPublisherURL=https://github.com/Sususu-slac/BL62_Reconstruction_GUI
DefaultDirName={localappdata}\Programs\BL62_Reconstruction_GUI
DefaultGroupName=BL62 Reconstruction
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir=release
OutputBaseFilename=BL62_Reconstruction_GUI_Setup_1.0.1
SetupIconFile=BL62_Recon.ico
UninstallDisplayIcon={app}\BL62_Reconstruction_GUI.exe
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
CloseApplications=yes
[Tasks]
Name: desktopicon; Description: "Create a desktop shortcut"; Flags: checkedonce
[Files]
Source: "dist\BL62_Reconstruction_GUI\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
[Icons]
Name: "{autoprograms}\BL62 Reconstruction"; Filename: "{app}\BL62_Reconstruction_GUI.exe"
Name: "{autodesktop}\BL62 Reconstruction"; Filename: "{app}\BL62_Reconstruction_GUI.exe"; Tasks: desktopicon
Name: "{autoprograms}\BL62 Reconstruction - Install Environments"; Filename: "{app}\_internal\install_windows.bat"; WorkingDir: "{app}\_internal"
[Run]
Filename: "{app}\_internal\install_windows.bat"; Description: "Prepare the reconstruction and SAM3 environments"; Flags: postinstall skipifsilent unchecked
Filename: "{app}\BL62_Reconstruction_GUI.exe"; Description: "Launch BL62 Reconstruction"; Flags: nowait postinstall skipifsilent
