#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <QFileDialog>
#include <QMessageBox>

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent), ui(new Ui::MainWindow), process(new QProcess(this)) {
    ui->setupUi(this);
    statusBar = new QStatusBar(this);
    setStatusBar(statusBar);

    // Establecer estilo con QStyleSheet
    this->setStyleSheet(
        "QPushButton { background-color: #4CAF50; color: white; padding: 5px; }"
        "QLineEdit { padding: 5px; }"
        "QTextEdit { background-color: #f0f0f0; }"
        );

    // Añadir opciones de versión de Python
    ui->pythonVersionComboBox->addItem("python");
    ui->pythonVersionComboBox->addItem("python3");

    // Conectar señales de QProcess para manejar la salida
    connect(process, &QProcess::readyReadStandardOutput, this, [=]() {
        ui->outputTextEdit->append(process->readAllStandardOutput());
    });
    connect(process, &QProcess::readyReadStandardError, this, [=]() {
        ui->outputTextEdit->append(process->readAllStandardError());
    });
    // Conectar acciones del menú y barra de herramientas
    connect(ui->actionBuscar, &QAction::triggered, this, &MainWindow::on_actionBuscar_triggered);
    connect(ui->actionEjecutar, &QAction::triggered, this, &MainWindow::on_actionEjecutar_triggered);
    connect(ui->actionSalir, &QAction::triggered, this, &MainWindow::on_actionSalir_triggered);
    connect(ui->actionAcercaDe, &QAction::triggered, this, &MainWindow::on_actionAcercaDe_triggered);


    // Conectar señal de errorOccurred para manejar errores
    connect(process, &QProcess::errorOccurred, this, [=](QProcess::ProcessError error) {
        QString errorMessage;
        switch (error) {
        case QProcess::FailedToStart:
            errorMessage = "El script no pudo iniciarse. Verifique la ruta al archivo y si Python está instalado correctamente.";
            break;
        case QProcess::Crashed:
            errorMessage = "El script se cerró inesperadamente.";
            break;
        case QProcess::Timedout:
            errorMessage = "El script tardó demasiado tiempo en ejecutarse.";
            break;
        case QProcess::WriteError:
            errorMessage = "Error al escribir en el proceso.";
            break;
        case QProcess::ReadError:
            errorMessage = "Error al leer la salida del proceso.";
            break;
        default:
            errorMessage = "Ocurrió un error desconocido.";
            break;
        }
        QMessageBox::critical(this, "Error de Ejecución", errorMessage);
        statusBar->showMessage("Error durante la ejecución del script.");
    });

    // Conectar señal finished para mostrar un mensaje al finalizar la ejecución del script
    connect(process, QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished),
            this, [=](int exitCode, QProcess::ExitStatus exitStatus) {
                if (exitStatus == QProcess::NormalExit && exitCode == 0) {
                    statusBar->showMessage("Script ejecutado con éxito.");
                } else {
                    QString errorMessage = "El script terminó con errores. Código de salida: " + QString::number(exitCode);
                    QMessageBox::warning(this, "Ejecución Incompleta", errorMessage);
                    statusBar->showMessage("El script terminó con errores.");
                }
            });
}

MainWindow::~MainWindow() {
    delete ui;
    delete process;
}

void MainWindow::on_browseButton_clicked() {
    QString fileName = QFileDialog::getOpenFileName(this, "Seleccionar archivo Python", "", "Archivos Python (*.py)");
    if (!fileName.isEmpty()) {
        ui->fileLineEdit->setText(fileName);
    }
}

void MainWindow::on_runButton_clicked() {
    QString filePath = ui->fileLineEdit->text();
    QString params = ui->paramsLineEdit->text();
    QString pythonInterpreter = ui->pythonVersionComboBox->currentText();

    if (filePath.isEmpty()) {
        QMessageBox::warning(this, "Advertencia", "No se ha seleccionado un archivo válido.");
        return;
    }

    ui->outputTextEdit->clear();  // Limpiar la salida antes de ejecutar
    statusBar->showMessage("Ejecutando script...");

    process->start(pythonInterpreter, QStringList() << filePath << params.split(" "));
}
void MainWindow::on_actionBuscar_triggered() {
    // Esta función ejecuta la misma lógica que el botón "Buscar"
    QString fileName = QFileDialog::getOpenFileName(this, "Seleccionar archivo Python", "", "Archivos Python (*.py)");
    if (!fileName.isEmpty()) {
        ui->fileLineEdit->setText(fileName);
    }
}

void MainWindow::on_actionEjecutar_triggered() {
    // Esta función ejecuta la misma lógica que el botón "Ejecutar"
    on_runButton_clicked();
}

void MainWindow::on_actionSalir_triggered() {
    // Esta función cierra la aplicación
    close();
}

void MainWindow::on_actionAcercaDe_triggered() {
    // Mostrar información sobre la aplicación
    QMessageBox::about(this, "Acerca de Python Executor",
                       "PDI Executor v1.0b\n\nUna herramienta para seleccionar y ejecutar scripts Python, de esta manera automatizar el proceso de ejecución y simpleza, con interfaz gráfica desarrollada en Qt. KN-TECH");
}

