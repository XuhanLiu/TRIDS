@echo off
setlocal

if not defined CONDA_PREFIX (
    echo ERROR: Run: conda activate trids126
    exit /b 1
)

set "PATH=%CONDA_PREFIX%;%CONDA_PREFIX%\Library\bin;%CONDA_PREFIX%\Scripts;%PATH%"

set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if exist "%VSWHERE%" (
    for /f "usebackq delims=" %%i in (`"%VSWHERE%" -latest -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do (
        if exist "%%i\VC\Auxiliary\Build\vcvars64.bat" (
            call "%%i\VC\Auxiliary\Build\vcvars64.bat"
            goto :vcvars_ok
        )
    )
)
if exist "%CONDA_PREFIX%\etc\conda\activate.d\vs2022_compiler_vars.bat" (
    call "%CONDA_PREFIX%\etc\conda\activate.d\vs2022_compiler_vars.bat"
)
:vcvars_ok

set "BUILD_DIR=%~dp0..\..\build"
if not exist "%BUILD_DIR%" mkdir "%BUILD_DIR%"
cd /d "%BUILD_DIR%"

cmake ..\cmake\windows -G Ninja -DCMAKE_BUILD_TYPE=Release  -DBUILD_PYTHON=ON -DCMAKE_MAKE_PROGRAM="%CONDA_PREFIX%\Library\bin\ninja.exe"
if errorlevel 1 exit /b 1

cmake --build . -j 4
exit /b %ERRORLEVEL%
