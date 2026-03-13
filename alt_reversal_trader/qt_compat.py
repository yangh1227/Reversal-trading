from __future__ import annotations

try:
    from PyQt5.QtCore import QEvent, QPropertyAnimation, Qt, QThread, QTimer, QUrl, pyqtSignal as Signal
    from PyQt5.QtGui import QBrush, QColor
    from PyQt5.QtWebEngineWidgets import QWebEngineSettings, QWebEngineView
    from PyQt5.QtWidgets import (
        QAbstractItemView,
        QApplication,
        QCheckBox,
        QComboBox as _QComboBox,
        QDoubleSpinBox as _QDoubleSpinBox,
        QFormLayout,
        QGridLayout,
        QGraphicsOpacityEffect,
        QGroupBox,
        QHBoxLayout,
        QHeaderView,
        QLabel,
        QLineEdit,
        QMainWindow,
        QMessageBox,
        QPlainTextEdit,
        QProgressBar,
        QPushButton,
        QRadioButton,
        QScrollArea,
        QSpinBox as _QSpinBox,
        QSplitter,
        QTableWidget,
        QTableWidgetItem,
        QTabWidget,
        QVBoxLayout,
        QWidget,
    )

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
    from PyQt6.QtCore import QEvent, QPropertyAnimation, Qt, QThread, QTimer, QUrl, pyqtSignal as Signal
    from PyQt6.QtGui import QBrush, QColor
    from PyQt6.QtWebEngineCore import QWebEngineSettings
    from PyQt6.QtWebEngineWidgets import QWebEngineView
    from PyQt6.QtWidgets import (
        QAbstractItemView,
        QApplication,
        QCheckBox,
        QComboBox as _QComboBox,
        QDoubleSpinBox as _QDoubleSpinBox,
        QFormLayout,
        QGridLayout,
        QGraphicsOpacityEffect,
        QGroupBox,
        QHBoxLayout,
        QHeaderView,
        QLabel,
        QLineEdit,
        QMainWindow,
        QMessageBox,
        QPlainTextEdit,
        QProgressBar,
        QPushButton,
        QRadioButton,
        QScrollArea,
        QSpinBox as _QSpinBox,
        QSplitter,
        QTableWidget,
        QTableWidgetItem,
        QTabWidget,
        QVBoxLayout,
        QWidget,
    )

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
