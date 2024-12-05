/********************************************************************************
** Form generated from reading UI file 'mainwindow.ui'
**
** Created by: Qt User Interface Compiler version 5.15.16
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_MAINWINDOW_H
#define UI_MAINWINDOW_H

#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QComboBox>
#include <QtWidgets/QGridLayout>
#include <QtWidgets/QLabel>
#include <QtWidgets/QLineEdit>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QMenu>
#include <QtWidgets/QMenuBar>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QStatusBar>
#include <QtWidgets/QTextEdit>
#include <QtWidgets/QToolBar>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_MainWindow
{
public:
    QAction *actionBuscar;
    QAction *actionEjecutar;
    QAction *actionSalir;
    QAction *actionAcercaDe;
    QWidget *centralwidget;
    QVBoxLayout *verticalLayout;
    QGridLayout *gridLayout;
    QLabel *labelSelectFile;
    QLineEdit *fileLineEdit;
    QPushButton *browseButton;
    QComboBox *pythonVersionComboBox;
    QLineEdit *paramsLineEdit;
    QPushButton *runButton;
    QTextEdit *outputTextEdit;
    QMenuBar *menubar;
    QMenu *menuArchivo;
    QMenu *menuAyuda;
    QToolBar *mainToolBar;
    QStatusBar *statusbar;

    void setupUi(QMainWindow *MainWindow)
    {
        if (MainWindow->objectName().isEmpty())
            MainWindow->setObjectName(QString::fromUtf8("MainWindow"));
        MainWindow->resize(800, 600);
        actionBuscar = new QAction(MainWindow);
        actionBuscar->setObjectName(QString::fromUtf8("actionBuscar"));
        actionBuscar->setCheckable(false);
        actionBuscar->setAutoRepeat(false);
        actionEjecutar = new QAction(MainWindow);
        actionEjecutar->setObjectName(QString::fromUtf8("actionEjecutar"));
        actionEjecutar->setAutoRepeat(false);
        actionSalir = new QAction(MainWindow);
        actionSalir->setObjectName(QString::fromUtf8("actionSalir"));
        actionAcercaDe = new QAction(MainWindow);
        actionAcercaDe->setObjectName(QString::fromUtf8("actionAcercaDe"));
        centralwidget = new QWidget(MainWindow);
        centralwidget->setObjectName(QString::fromUtf8("centralwidget"));
        verticalLayout = new QVBoxLayout(centralwidget);
        verticalLayout->setObjectName(QString::fromUtf8("verticalLayout"));
        gridLayout = new QGridLayout();
        gridLayout->setObjectName(QString::fromUtf8("gridLayout"));
        labelSelectFile = new QLabel(centralwidget);
        labelSelectFile->setObjectName(QString::fromUtf8("labelSelectFile"));

        gridLayout->addWidget(labelSelectFile, 0, 0, 1, 1);

        fileLineEdit = new QLineEdit(centralwidget);
        fileLineEdit->setObjectName(QString::fromUtf8("fileLineEdit"));

        gridLayout->addWidget(fileLineEdit, 0, 1, 1, 1);

        browseButton = new QPushButton(centralwidget);
        browseButton->setObjectName(QString::fromUtf8("browseButton"));

        gridLayout->addWidget(browseButton, 0, 2, 1, 1);

        pythonVersionComboBox = new QComboBox(centralwidget);
        pythonVersionComboBox->setObjectName(QString::fromUtf8("pythonVersionComboBox"));

        gridLayout->addWidget(pythonVersionComboBox, 0, 3, 1, 1);

        paramsLineEdit = new QLineEdit(centralwidget);
        paramsLineEdit->setObjectName(QString::fromUtf8("paramsLineEdit"));

        gridLayout->addWidget(paramsLineEdit, 1, 0, 1, 3);

        runButton = new QPushButton(centralwidget);
        runButton->setObjectName(QString::fromUtf8("runButton"));

        gridLayout->addWidget(runButton, 2, 0, 1, 3);


        verticalLayout->addLayout(gridLayout);

        outputTextEdit = new QTextEdit(centralwidget);
        outputTextEdit->setObjectName(QString::fromUtf8("outputTextEdit"));
        outputTextEdit->setReadOnly(true);

        verticalLayout->addWidget(outputTextEdit);

        MainWindow->setCentralWidget(centralwidget);
        menubar = new QMenuBar(MainWindow);
        menubar->setObjectName(QString::fromUtf8("menubar"));
        menubar->setGeometry(QRect(0, 0, 800, 23));
        menuArchivo = new QMenu(menubar);
        menuArchivo->setObjectName(QString::fromUtf8("menuArchivo"));
        menuAyuda = new QMenu(menubar);
        menuAyuda->setObjectName(QString::fromUtf8("menuAyuda"));
        MainWindow->setMenuBar(menubar);
        mainToolBar = new QToolBar(MainWindow);
        mainToolBar->setObjectName(QString::fromUtf8("mainToolBar"));
        MainWindow->addToolBar(Qt::TopToolBarArea, mainToolBar);
        statusbar = new QStatusBar(MainWindow);
        statusbar->setObjectName(QString::fromUtf8("statusbar"));
        MainWindow->setStatusBar(statusbar);

        menubar->addAction(menuArchivo->menuAction());
        menubar->addAction(menuAyuda->menuAction());
        menuArchivo->addAction(actionBuscar);
        menuArchivo->addAction(actionEjecutar);
        menuArchivo->addAction(actionSalir);
        menuAyuda->addAction(actionAcercaDe);
        mainToolBar->addAction(actionBuscar);
        mainToolBar->addAction(actionEjecutar);

        retranslateUi(MainWindow);

        QMetaObject::connectSlotsByName(MainWindow);
    } // setupUi

    void retranslateUi(QMainWindow *MainWindow)
    {
        MainWindow->setWindowTitle(QCoreApplication::translate("MainWindow", "PDI Executor - KN-TECH", nullptr));
        actionBuscar->setText(QCoreApplication::translate("MainWindow", "Buscar Archivo", nullptr));
#if QT_CONFIG(shortcut)
        actionBuscar->setShortcut(QCoreApplication::translate("MainWindow", "Ctrl+O", nullptr));
#endif // QT_CONFIG(shortcut)
        actionEjecutar->setText(QCoreApplication::translate("MainWindow", "Ejecutar Script", nullptr));
#if QT_CONFIG(shortcut)
        actionEjecutar->setShortcut(QCoreApplication::translate("MainWindow", "Ctrl+R", nullptr));
#endif // QT_CONFIG(shortcut)
        actionSalir->setText(QCoreApplication::translate("MainWindow", "Salir", nullptr));
#if QT_CONFIG(shortcut)
        actionSalir->setShortcut(QCoreApplication::translate("MainWindow", "Ctrl+Q", nullptr));
#endif // QT_CONFIG(shortcut)
        actionAcercaDe->setText(QCoreApplication::translate("MainWindow", "Acerca de...", nullptr));
        labelSelectFile->setText(QCoreApplication::translate("MainWindow", "Archivo Python:", nullptr));
        browseButton->setText(QCoreApplication::translate("MainWindow", "Buscar", nullptr));
#if QT_CONFIG(tooltip)
        pythonVersionComboBox->setToolTip(QCoreApplication::translate("MainWindow", "Seleccionar versi\303\263n de Python", nullptr));
#endif // QT_CONFIG(tooltip)
        paramsLineEdit->setPlaceholderText(QCoreApplication::translate("MainWindow", "Par\303\241metros del script (opcional)", nullptr));
        runButton->setText(QCoreApplication::translate("MainWindow", "Ejecutar Script", nullptr));
        outputTextEdit->setPlaceholderText(QCoreApplication::translate("MainWindow", "Salida del script...", nullptr));
        menuArchivo->setTitle(QCoreApplication::translate("MainWindow", "Archivo", nullptr));
        menuAyuda->setTitle(QCoreApplication::translate("MainWindow", "Ayuda", nullptr));
        mainToolBar->setWindowTitle(QCoreApplication::translate("MainWindow", "Herramientas", nullptr));
    } // retranslateUi

};

namespace Ui {
    class MainWindow: public Ui_MainWindow {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_MAINWINDOW_H
