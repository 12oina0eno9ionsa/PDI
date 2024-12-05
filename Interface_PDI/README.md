# PDI Executor - GUI with Qt

**Version: 1.0b**  
**Author: KN-TECH**

## Overview

**PDI Executor** is a graphical tool developed with **Qt** that allows users to select and execute **the Python project** script about the models, with the ability to view the output directly within the interface. The main objective of this application is to simplify and automate the process of running this scripts, making it more user-friendly, especially for those who prefer not to use the command line. This tool is ideal for developers, researchers, and other users seeking a quick and efficient way to execute model scripts from a convenient graphical interface.

If you know how to run a project using **Qt** you can skip these explanations, is just simply Build -> then Run.

Verify that al the contents of the Project PDI is on the **same directory** as the python script.

## Table of Contents

- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Project Setup](#project-setup)
- [Running the Application](#running-the-application)
- [Features](#features)
- [Usage](#usage)
- [Troubleshooting](#troubleshooting)
- [License](#license)

## System Requirements

To run **PDI Executor**, you will need the following:

### **Supported Operating Systems**:
- **Windows** (7, 8, 10, 11) (Need to be verified)
- **Linux** (Ubuntu-based distributions or Arch Linux, such as Manjaro)
- **macOS** (10.14 Mojave and higher, stability needs to be verified)

### **Dependencies**:
- **Qt 5.15 or later** (including **Qt Creator** to build the application).
- **Python 3.x** (must be installed and set up in the system PATH).
- **C++ Compiler**:
  - **Windows**: **MinGW** (included with Qt).
  - **Linux**: **GCC**.

### **Additional Libraries**:
- **PDI Executor** requires that `python3` or `python` is accessible from the command line. Make sure the executable is included in the system's PATH for proper functionality.

## Installation

### **Windows** (Not Recommended):

1. Download and install **Qt** from [https://www.qt.io/download](https://www.qt.io/download).
2. During installation, ensure to include **MinGW** and the **Qt Widgets** module.
3. Download and install **Python** from [https://www.python.org/downloads/](https://www.python.org/downloads/). Make sure to check **"Add Python to PATH"** during installation.

### **Linux (Ubuntu / Arch Linux / Manjaro)**:

1. Update repositories and install **Qt**:
   - **Ubuntu**:
     ```sh
     sudo apt update
     sudo apt install qt5-default qtcreator build-essential
     ```
   - **Arch / Manjaro**:
     ```sh
     sudo pacman -Syu
     sudo pacman -S qt5-base qtcreator base-devel
     ```

2. Install **Python** (if not already installed):
   ```sh
   sudo apt install python3  # Ubuntu
   sudo pacman -S python     # Arch/Manjaro
   ```

## Project Setup

1. Open **Qt Creator** and select `File > Open Project...`.
2. Navigate to the folder containing the `.pro` file of the project.
3. Select the appropriate **build kit** (e.g., `Desktop Qt 5.15.2 MinGW 64-bit`).
4. **Build** the project (`Ctrl + B`) to ensure that all dependencies are correctly set up.

## Running the Application

### **Windows**:

- Once the project is built in **Qt Creator**, navigate to the build folder (`build-PDIExecutor-Desktop_Qt_...`) and run the **PDIExecutor.exe** file.

### **Linux**:

- From **Qt Creator**, click **Run** (`Ctrl + R`) to launch the application.
- Alternatively, navigate to the build folder and run the executable:
  ```sh
  ./PDIExecutor
  ```

## Features

1. **Python File Selection**:
   - You can select any local `.py` file using the file dialog.

2. **Script Execution**:
   - Click the **Run** button to execute the selected Python script, and the output will be displayed within the interface.

3. **Input Parameters**:
   - Optional **parameters** can be entered, which will be passed to the Python script during execution.

4. **Support for Multiple Python Versions**:
   - Choose between `python` and `python3`, which is useful for systems with multiple versions of Python installed.

5. **Toolbar and Menu**:
   - **Toolbar** for quick actions like **Browse File** and **Run Script**.
   - **File Menu**: Includes options like **Browse**, **Run**, and **Exit**.
   - **Help Menu**: Includes an **About** option to display information about the software.

6. **Status Bar**:
   - Displays messages about the script execution status (e.g., "Running script..." or "Script executed successfully").

## Usage

1. **Browse Python File**:
   - Click **Browse** (or select **File > Browse File** from the menu) to choose the Python file (`.py`) you wish to run.

2. **Enter Parameters (Optional)**:
   - If the Python script requires **parameters**, enter them in the provided field before executing.

3. **Run the Script**:
   - Click **Run** (or select **File > Run Script** from the menu) to execute the script.
   - The **output** of the script will be displayed in the text area below.

4. **Close the Application**:
   - Select **File > Exit** or use the shortcut `Ctrl + Q` to close the application.

5. **About the Application**:
   - Select **Help > About...** to view version information and details about the software.

## Troubleshooting

- **Issue**: The application cannot find `python`.
  - **Solution**: Ensure **Python** is correctly installed and added to the **PATH** of the system. On Linux, you can test with:
    ```sh
    which python3
    ```
  - On **Windows**, check the **Environment Variables** to confirm that the Python directory is included in the PATH.

- **Issue**: The application displays `"Failed to start script"` when attempting to run.
  - **Solution**: Ensure the selected file is a valid Python script. Verify that the file has the correct **execution permissions** (on Linux, you can use `chmod +x script_name.py`).

- **Issue**: Clicking "Browse" or "Run" opens two windows.
  - **Solution**: This was caused by duplicate connections. Make sure to use `Qt::UniqueConnection` in the menu and toolbar connections, as detailed in the provided solution.

## License

This software is licensed under the **MIT License**, meaning you are free to use, modify, and distribute this code, provided the original license is included.