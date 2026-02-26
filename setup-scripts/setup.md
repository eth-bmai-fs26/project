# General setup scripts

In this folder there are two different scripts that are supposed to be used by participants that do not have yet python installed. Both scripts check for python to be installed, create a virtual enviorment using venv and download the requirements from `requirements.txt`.

There are two scripts one for MacOS and one for Windows.

### MacOS

Make the script executable.
`chmod +x setup.sh`

Run the setup
`./setup.sh`

### Windows

This script is made to be run in a powershell terminal.
Make the script executable.
`Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
   .\setup.ps1`

Run the setup
`.\setup.ps1`