#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QProcess>

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class MainWindow : public QMainWindow {
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private slots:
    void on_browseButton_clicked();
    void on_runButton_clicked();
    void on_actionBuscar_triggered();
    void on_actionEjecutar_triggered();
    void on_actionSalir_triggered();
    void on_actionAcercaDe_triggered();

private:
    Ui::MainWindow *ui;
    QProcess *process;
    QStatusBar *statusBar;
};

#endif // MAINWINDOW_H
