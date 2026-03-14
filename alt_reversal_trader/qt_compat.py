from __future__ import annotations

from importlib import import_module


def _load_pyqt5_bindings():
    qtcore = import_module("PyQt5.QtCore")
    qtgui = import_module("PyQt5.QtGui")
    qtwebengine = import_module("PyQt5.QtWebEngineWidgets")
    qtwidgets = import_module("PyQt5.QtWidgets")
    return qtcore, qtgui, qtwebengine, qtwidgets


def _load_pyqt6_bindings():
    qtcore = import_module("PyQt6.QtCore")
    qtgui = import_module("PyQt6.QtGui")
    qtwebenginecore = import_module("PyQt6.QtWebEngineCore")
    qtwebenginewidgets = import_module("PyQt6.QtWebEngineWidgets")
    qtwidgets = import_module("PyQt6.QtWidgets")
    return qtcore, qtgui, qtwebenginecore, qtwebenginewidgets, qtwidgets


try:
    _qtcore, _qtgui, _qtwebenginewidgets, _qtwidgets = _load_pyqt5_bindings()

    QEvent = _qtcore.QEvent
    QPropertyAnimation = _qtcore.QPropertyAnimation
    Qt = _qtcore.Qt
    QThread = _qtcore.QThread
    QTimer = _qtcore.QTimer
    QUrl = _qtcore.QUrl
    Signal = _qtcore.pyqtSignal
    QBrush = _qtgui.QBrush
    QColor = _qtgui.QColor
    QWebEngineSettings = _qtwebenginewidgets.QWebEngineSettings
    QWebEngineView = _qtwebenginewidgets.QWebEngineView
    QAbstractItemView = _qtwidgets.QAbstractItemView
    QApplication = _qtwidgets.QApplication
    QCheckBox = _qtwidgets.QCheckBox
    _QComboBox = _qtwidgets.QComboBox
    _QDoubleSpinBox = _qtwidgets.QDoubleSpinBox
    QFormLayout = _qtwidgets.QFormLayout
    QGridLayout = _qtwidgets.QGridLayout
    QGraphicsOpacityEffect = _qtwidgets.QGraphicsOpacityEffect
    QGroupBox = _qtwidgets.QGroupBox
    QHBoxLayout = _qtwidgets.QHBoxLayout
    QHeaderView = _qtwidgets.QHeaderView
    QLabel = _qtwidgets.QLabel
    QLineEdit = _qtwidgets.QLineEdit
    QMainWindow = _qtwidgets.QMainWindow
    QMessageBox = _qtwidgets.QMessageBox
    QPlainTextEdit = _qtwidgets.QPlainTextEdit
    QProgressBar = _qtwidgets.QProgressBar
    QPushButton = _qtwidgets.QPushButton
    QRadioButton = _qtwidgets.QRadioButton
    QScrollArea = _qtwidgets.QScrollArea
    _QSpinBox = _qtwidgets.QSpinBox
    QSplitter = _qtwidgets.QSplitter
    QTableWidget = _qtwidgets.QTableWidget
    QTableWidgetItem = _qtwidgets.QTableWidgetItem
    QTabWidget = _qtwidgets.QTabWidget
    QVBoxLayout = _qtwidgets.QVBoxLayout
    QWidget = _qtwidgets.QWidget

    QT_API = "PyQt5"

    def app_exec(app: QApplication) -> int:
        return app.exec_()

    HORIZONTAL = Qt.Horizontal
    VERTICAL = Qt.Vertical
    ALIGN_LEFT = Qt.AlignLeft
    USER_ROLE = Qt.UserRole
    PASSWORD_ECHO = QLineEdit.Password
    SELECT_ROWS = QAbstractItemView.SelectRows
    SINGLE_SELECTION = QAbstractItemView.SingleSelection
    NO_EDIT_TRIGGERS = QAbstractItemView.NoEditTriggers
    WEB_ATTR_FILE_URLS = QWebEngineSettings.LocalContentCanAccessFileUrls
    EVENT_KEY_PRESS = QEvent.KeyPress
    KEY_UP = Qt.Key_Up
    KEY_DOWN = Qt.Key_Down

except ImportError:
    _qtcore, _qtgui, _qtwebenginecore, _qtwebenginewidgets, _qtwidgets = _load_pyqt6_bindings()

    QEvent = _qtcore.QEvent
    QPropertyAnimation = _qtcore.QPropertyAnimation
    Qt = _qtcore.Qt
    QThread = _qtcore.QThread
    QTimer = _qtcore.QTimer
    QUrl = _qtcore.QUrl
    Signal = _qtcore.pyqtSignal
    QBrush = _qtgui.QBrush
    QColor = _qtgui.QColor
    QWebEngineSettings = _qtwebenginecore.QWebEngineSettings
    QWebEngineView = _qtwebenginewidgets.QWebEngineView
    QAbstractItemView = _qtwidgets.QAbstractItemView
    QApplication = _qtwidgets.QApplication
    QCheckBox = _qtwidgets.QCheckBox
    _QComboBox = _qtwidgets.QComboBox
    _QDoubleSpinBox = _qtwidgets.QDoubleSpinBox
    QFormLayout = _qtwidgets.QFormLayout
    QGridLayout = _qtwidgets.QGridLayout
    QGraphicsOpacityEffect = _qtwidgets.QGraphicsOpacityEffect
    QGroupBox = _qtwidgets.QGroupBox
    QHBoxLayout = _qtwidgets.QHBoxLayout
    QHeaderView = _qtwidgets.QHeaderView
    QLabel = _qtwidgets.QLabel
    QLineEdit = _qtwidgets.QLineEdit
    QMainWindow = _qtwidgets.QMainWindow
    QMessageBox = _qtwidgets.QMessageBox
    QPlainTextEdit = _qtwidgets.QPlainTextEdit
    QProgressBar = _qtwidgets.QProgressBar
    QPushButton = _qtwidgets.QPushButton
    QRadioButton = _qtwidgets.QRadioButton
    QScrollArea = _qtwidgets.QScrollArea
    _QSpinBox = _qtwidgets.QSpinBox
    QSplitter = _qtwidgets.QSplitter
    QTableWidget = _qtwidgets.QTableWidget
    QTableWidgetItem = _qtwidgets.QTableWidgetItem
    QTabWidget = _qtwidgets.QTabWidget
    QVBoxLayout = _qtwidgets.QVBoxLayout
    QWidget = _qtwidgets.QWidget

    QT_API = "PyQt6"

    def app_exec(app: QApplication) -> int:
        return app.exec()

    HORIZONTAL = Qt.Orientation.Horizontal
    VERTICAL = Qt.Orientation.Vertical
    ALIGN_LEFT = Qt.AlignmentFlag.AlignLeft
    USER_ROLE = Qt.ItemDataRole.UserRole
    PASSWORD_ECHO = QLineEdit.EchoMode.Password
    SELECT_ROWS = QAbstractItemView.SelectionBehavior.SelectRows
    SINGLE_SELECTION = QAbstractItemView.SelectionMode.SingleSelection
    NO_EDIT_TRIGGERS = QAbstractItemView.EditTrigger.NoEditTriggers
    WEB_ATTR_FILE_URLS = QWebEngineSettings.WebAttribute.LocalContentCanAccessFileUrls
    EVENT_KEY_PRESS = QEvent.Type.KeyPress
    KEY_UP = Qt.Key.Key_Up
    KEY_DOWN = Qt.Key.Key_Down


class QSpinBox(_QSpinBox):
    def wheelEvent(self, event):  # type: ignore[override]
        event.ignore()


class QDoubleSpinBox(_QDoubleSpinBox):
    def wheelEvent(self, event):  # type: ignore[override]
        event.ignore()


class QComboBox(_QComboBox):
    def wheelEvent(self, event):  # type: ignore[override]
        event.ignore()


__all__ = [
    "ALIGN_LEFT",
    "EVENT_KEY_PRESS",
    "HORIZONTAL",
    "KEY_DOWN",
    "KEY_UP",
    "PASSWORD_ECHO",
    "QT_API",
    "NO_EDIT_TRIGGERS",
    "SELECT_ROWS",
    "SINGLE_SELECTION",
    "Signal",
    "Qt",
    "QEvent",
    "USER_ROLE",
    "VERTICAL",
    "WEB_ATTR_FILE_URLS",
    "QApplication",
    "QBrush",
    "QCheckBox",
    "QColor",
    "QComboBox",
    "QDoubleSpinBox",
    "QFormLayout",
    "QGraphicsOpacityEffect",
    "QGridLayout",
    "QGroupBox",
    "QHBoxLayout",
    "QHeaderView",
    "QLabel",
    "QLineEdit",
    "QMainWindow",
    "QMessageBox",
    "QPlainTextEdit",
    "QProgressBar",
    "QPropertyAnimation",
    "QPushButton",
    "QRadioButton",
    "QScrollArea",
    "QSpinBox",
    "QSplitter",
    "QTableWidget",
    "QTableWidgetItem",
    "QTabWidget",
    "QThread",
    "QTimer",
    "QUrl",
    "QWebEngineView",
    "QVBoxLayout",
    "QWidget",
    "app_exec",
]
