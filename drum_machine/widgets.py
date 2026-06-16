import numpy as np

from PySide6.QtCore import QPointF, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QSizePolicy, QToolButton, QToolTip, QWidget

from .config import SAMPLE_RATE


class StepButton(QToolButton):
    rightClicked = Signal(int, int)

    def __init__(self, track: int, step: int):
        super().__init__()
        self.track = track
        self.step = step
        self.setCheckable(True)
        self.setMinimumSize(28, 32)
        self.setMaximumHeight(42)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setText(str(step + 1))
        self.setToolTip(
            f"Left-click toggles step {step + 1} for track {track + 1}. Right-click selects it for editing without toggling."
        )

    def mousePressEvent(self, event):
        if event.button() == Qt.RightButton:
            self.rightClicked.emit(self.track, self.step)
            event.accept()
            return
        super().mousePressEvent(event)


class EnvelopeEditor(QWidget):
    def __init__(self, points, on_change):
        super().__init__()
        self.points = [list(point) for point in points]
        self.on_change = on_change
        self.drag_index: int | None = None
        self.setMinimumSize(220, 96)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setMouseTracking(True)

    def set_points(self, points):
        self.points = [list(point) for point in points]
        self.update()

    def paintEvent(self, event):
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        rect = self.rect().adjusted(10, 10, -10, -10)

        painter.fillRect(self.rect(), QColor("#15191f"))
        painter.setPen(QPen(QColor("#2f3946"), 1))
        for index in range(5):
            x = rect.left() + rect.width() * index / 4
            y = rect.top() + rect.height() * index / 4
            painter.drawLine(int(x), rect.top(), int(x), rect.bottom())
            painter.drawLine(rect.left(), int(y), rect.right(), int(y))

        screen_points = [self._to_screen(point, rect) for point in self.points]
        painter.setPen(QPen(QColor("#58c4dd"), 3))
        for start, end in zip(screen_points, screen_points[1:]):
            painter.drawLine(start, end)

        painter.setPen(QPen(QColor("#f2d16b"), 2))
        painter.setBrush(QColor("#20242a"))
        for point in screen_points:
            painter.drawEllipse(point, 6, 6)

    def mousePressEvent(self, event):
        self.drag_index = self._nearest_point(event.position())
        self._move_point(event.position())

    def mouseMoveEvent(self, event):
        if self.drag_index is not None:
            self._move_point(event.position())

    def mouseReleaseEvent(self, event):
        del event
        self.drag_index = None

    def _nearest_point(self, position: QPointF) -> int:
        rect = self.rect().adjusted(10, 10, -10, -10)
        screen_points = [self._to_screen(point, rect) for point in self.points]
        distances = [
            (point.x() - position.x()) ** 2 + (point.y() - position.y()) ** 2
            for point in screen_points
        ]
        return int(min(range(len(distances)), key=distances.__getitem__))

    def _move_point(self, position: QPointF):
        if self.drag_index is None:
            return
        rect = self.rect().adjusted(10, 10, -10, -10)
        x = (position.x() - rect.left()) / max(1, rect.width())
        y = 1.0 - (position.y() - rect.top()) / max(1, rect.height())
        x = max(0.0, min(1.0, x))
        y = max(0.0, min(1.0, y))
        if self.drag_index == 0:
            x = 0.0
        elif self.drag_index == len(self.points) - 1:
            x = 1.0
        else:
            left = self.points[self.drag_index - 1][0] + 0.01
            right = self.points[self.drag_index + 1][0] - 0.01
            x = max(left, min(right, x))
        self.points[self.drag_index] = [round(x, 3), round(y, 3)]
        self.update()
        self.on_change([list(point) for point in self.points])

    def _to_screen(self, point, rect) -> QPointF:
        return QPointF(
            rect.left() + point[0] * rect.width(),
            rect.top() + (1.0 - point[1]) * rect.height(),
        )


class WaveformPreview(QWidget):
    def __init__(self):
        super().__init__()
        self.samples = np.zeros(0, dtype=np.float32)
        self.setMinimumHeight(64)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setToolTip("Preview of the current patch waveform.")

    def set_audio(self, audio: np.ndarray):
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        if len(audio) > 1200:
            positions = np.linspace(0, len(audio) - 1, 1200).astype(int)
            audio = audio[positions]
        peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
        self.samples = (audio / peak).astype(np.float32) if peak > 0.0 else np.zeros(0, dtype=np.float32)
        self.update()

    def paintEvent(self, event):
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        rect = self.rect().adjusted(8, 6, -8, -6)
        painter.fillRect(self.rect(), QColor("#15191f"))
        painter.setPen(QPen(QColor("#2f3946"), 1))
        center_y = rect.center().y()
        painter.drawLine(rect.left(), center_y, rect.right(), center_y)
        for index in range(1, 4):
            x = rect.left() + rect.width() * index / 4
            painter.drawLine(int(x), rect.top(), int(x), rect.bottom())

        if len(self.samples) < 2:
            painter.setPen(QPen(QColor("#6f7c8e"), 1))
            painter.drawText(rect, Qt.AlignCenter, "No preview")
            return

        painter.setPen(QPen(QColor("#58c4dd"), 2))
        previous = None
        for index, sample in enumerate(self.samples):
            x = rect.left() + rect.width() * index / max(1, len(self.samples) - 1)
            y = center_y - sample * rect.height() * 0.42
            point = QPointF(x, y)
            if previous is not None:
                painter.drawLine(previous, point)
            previous = point


class LevelProfilePreview(QWidget):
    def __init__(self):
        super().__init__()
        self.levels = np.zeros(0, dtype=np.float32)
        self.peak = 0.0
        self.rms = 0.0
        self.setMinimumHeight(64)
        self.setMinimumWidth(180)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setToolTip("Shows the rendered patch level over time.")

    def set_audio(self, audio: np.ndarray):
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        audio = audio.astype(np.float32, copy=False)
        if len(audio) == 0:
            self.levels = np.zeros(0, dtype=np.float32)
            self.peak = 0.0
            self.rms = 0.0
            self.update()
            return

        self.peak = float(np.max(np.abs(audio)))
        self.rms = float(np.sqrt(np.mean(audio * audio)))
        bucket_count = 36
        edges = np.linspace(0, len(audio), bucket_count + 1).astype(int)
        levels = []
        for start, end in zip(edges[:-1], edges[1:]):
            segment = audio[start:max(start + 1, end)]
            levels.append(float(np.sqrt(np.mean(segment * segment))))
        levels = np.asarray(levels, dtype=np.float32)
        peak = float(np.max(levels)) if len(levels) else 0.0
        self.levels = levels / peak if peak > 0.0 else np.zeros(0, dtype=np.float32)
        self.update()

    def paintEvent(self, event):
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        rect = self.rect().adjusted(8, 6, -8, -6)
        painter.fillRect(self.rect(), QColor("#15191f"))
        painter.setPen(QPen(QColor("#2f3946"), 1))
        painter.drawLine(rect.left(), rect.bottom(), rect.right(), rect.bottom())

        painter.setPen(QPen(QColor("#b8c4d2"), 1))
        painter.drawText(rect.adjusted(2, 0, -2, -32), Qt.AlignLeft | Qt.AlignTop, "Level")
        painter.drawText(
            rect.adjusted(2, 0, -2, -32),
            Qt.AlignRight | Qt.AlignTop,
            f"P {self.peak:.2f}  R {self.rms:.2f}",
        )

        if len(self.levels) == 0:
            painter.setPen(QPen(QColor("#6f7c8e"), 1))
            painter.drawText(rect, Qt.AlignCenter, "No level")
            return

        graph_rect = rect.adjusted(0, 18, 0, 0)
        gap = 2
        bar_width = max(2, (graph_rect.width() - gap * (len(self.levels) - 1)) / max(1, len(self.levels)))
        for index, level in enumerate(self.levels):
            x = graph_rect.left() + index * (bar_width + gap)
            height = max(2, level * graph_rect.height())
            top = graph_rect.bottom() - height
            color = QColor("#f2d16b") if index < 5 else QColor("#58c4dd")
            painter.fillRect(int(x), int(top), int(bar_width), int(height), color)


class SpectrumPreview(QWidget):
    def __init__(self):
        super().__init__()
        self.bands = np.zeros(0, dtype=np.float32)
        self.setMinimumHeight(64)
        self.setMinimumWidth(180)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setToolTip("Shows the rendered patch frequency balance.")

    def set_audio(self, audio: np.ndarray):
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        audio = audio.astype(np.float32, copy=False)
        if len(audio) < 8:
            self.bands = np.zeros(0, dtype=np.float32)
            self.update()
            return

        count = min(8192, len(audio))
        segment = audio[:count] * np.hanning(count).astype(np.float32)
        magnitudes = np.abs(np.fft.rfft(segment))
        frequencies = np.fft.rfftfreq(count, 1.0 / SAMPLE_RATE)
        edges = np.geomspace(35.0, 18000.0, 33)
        bands = []
        for low, high in zip(edges[:-1], edges[1:]):
            mask = (frequencies >= low) & (frequencies < high)
            bands.append(float(np.mean(magnitudes[mask])) if np.any(mask) else 0.0)
        bands = np.asarray(bands, dtype=np.float32)
        if len(bands):
            bands = np.log1p(bands)
        peak = float(np.max(bands)) if len(bands) else 0.0
        self.bands = bands / peak if peak > 0.0 else np.zeros(0, dtype=np.float32)
        self.update()

    def paintEvent(self, event):
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        rect = self.rect().adjusted(8, 6, -8, -6)
        painter.fillRect(self.rect(), QColor("#15191f"))
        painter.setPen(QPen(QColor("#b8c4d2"), 1))
        painter.drawText(rect.adjusted(2, 0, -2, -32), Qt.AlignLeft | Qt.AlignTop, "Spectrum")
        painter.setPen(QPen(QColor("#2f3946"), 1))
        for index in range(1, 4):
            y = rect.top() + rect.height() * index / 4
            painter.drawLine(rect.left(), int(y), rect.right(), int(y))

        if len(self.bands) == 0:
            painter.setPen(QPen(QColor("#6f7c8e"), 1))
            painter.drawText(rect, Qt.AlignCenter, "No spectrum")
            return

        graph_rect = rect.adjusted(0, 18, 0, 0)
        gap = 2
        bar_width = max(2, (graph_rect.width() - gap * (len(self.bands) - 1)) / max(1, len(self.bands)))
        for index, band in enumerate(self.bands):
            x = graph_rect.left() + index * (bar_width + gap)
            height = max(2, band * graph_rect.height())
            top = graph_rect.bottom() - height
            color = QColor("#9fe3bf") if index < len(self.bands) * 0.35 else QColor("#58c4dd")
            painter.fillRect(int(x), int(top), int(bar_width), int(height), color)


class OutputSpectrumMeter(QWidget):
    def __init__(self):
        super().__init__()
        self.bands = np.zeros(24, dtype=np.float32)
        self.setFixedSize(176, 36)
        self.setToolTip("Shows the final output spectrum after master processing.")

    def set_bands(self, bands: np.ndarray):
        self.bands = np.asarray(bands, dtype=np.float32)
        self.update()

    def paintEvent(self, event):
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor("#15191f"))
        rect = self.rect().adjusted(6, 5, -6, -5)
        painter.setPen(QPen(QColor("#2f3946"), 1))
        for index in range(1, 4):
            y = rect.top() + rect.height() * index / 4
            painter.drawLine(rect.left(), int(y), rect.right(), int(y))

        if len(self.bands) == 0:
            return
        gap = 2
        bar_width = max(2, (rect.width() - gap * (len(self.bands) - 1)) / max(1, len(self.bands)))
        for index, band in enumerate(self.bands):
            level = float(np.clip(band, 0.0, 1.0))
            x = rect.left() + index * (bar_width + gap)
            height = max(1, level * rect.height())
            top = rect.bottom() - height
            if index < len(self.bands) * 0.35:
                color = QColor("#9fe3bf")
            elif index < len(self.bands) * 0.75:
                color = QColor("#58c4dd")
            else:
                color = QColor("#f2d16b")
            painter.fillRect(int(x), int(top), int(bar_width), int(height), color)


class ChannelSpectrumDisplay(QWidget):
    def __init__(self, channel_name: str):
        super().__init__()
        self.channel_name = channel_name
        self.bands_db = np.full(56, -72.0, dtype=np.float32)
        self.average_bands_db = self.bands_db.copy()
        self.average_window = 12
        self.average_history: list[np.ndarray] = []
        self.total_db = -72.0
        self.selected_index = 24
        self.edges = np.geomspace(25.0, SAMPLE_RATE * 0.45, len(self.bands_db) + 1)
        self.setMinimumHeight(260)
        self.setMouseTracking(True)
        self.setToolTip("Live output spectrum for this channel. Green bars are current level; the magenta trace is averaged.")

    def set_spectrum(self, bands_db: np.ndarray, total_db: float):
        self.bands_db = np.asarray(bands_db, dtype=np.float32)
        self.total_db = float(total_db)
        if len(self.edges) != len(self.bands_db) + 1:
            self.edges = np.geomspace(25.0, SAMPLE_RATE * 0.45, len(self.bands_db) + 1)
            self.selected_index = min(self.selected_index, len(self.bands_db) - 1)
            self.average_history.clear()
            self.average_bands_db = self.bands_db.copy()
        self._add_average_frame()
        self.update()

    def set_average_window(self, value: int):
        self.average_window = int(np.clip(value, 1, 128))
        self.average_history = self.average_history[-self.average_window :]
        self._update_average_trace()
        self.update()

    def _add_average_frame(self):
        if len(self.bands_db) == 0:
            self.average_history.clear()
            self.average_bands_db = self.bands_db.copy()
            return
        linear = np.power(10.0, self.bands_db / 20.0).astype(np.float32)
        self.average_history.append(linear)
        self.average_history = self.average_history[-self.average_window :]
        self._update_average_trace()

    def _update_average_trace(self):
        if not self.average_history:
            self.average_bands_db = self.bands_db.copy()
            return
        history = [frame for frame in self.average_history if len(frame) == len(self.bands_db)]
        if not history:
            self.average_history.clear()
            self.average_bands_db = self.bands_db.copy()
            return
        averaged = np.mean(np.stack(history, axis=0), axis=0)
        averaged_db = 20.0 * np.log10(np.maximum(averaged, 1.0e-6))
        if len(averaged_db) >= 3:
            padded = np.pad(averaged_db, (1, 1), mode="edge")
            averaged_db = (
                padded[:-2] * 0.22
                + padded[1:-1] * 0.56
                + padded[2:] * 0.22
            )
        self.average_bands_db = np.clip(averaged_db, -72.0, 6.0).astype(np.float32)

    def paintEvent(self, event):
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor("#050607"))

        graph = self._graph_rect()
        min_db = -72.0
        max_db = 6.0

        selected = int(np.clip(self.selected_index, 0, max(0, len(self.bands_db) - 1)))
        selected_freq = float(np.sqrt(self.edges[selected] * self.edges[selected + 1]))

        painter.setPen(QPen(QColor("#9aa3ad"), 1))
        painter.drawRect(graph)
        for db in (-60, -48, -36, -24, -12, 0):
            y = self._db_to_y(db, graph, min_db, max_db)
            painter.setPen(QPen(QColor("#2d333a"), 1))
            painter.drawLine(graph.left(), y, graph.right(), y)
            painter.setPen(QPen(QColor("#c6ccd3"), 1))
            painter.drawText(4, y + 4, f"{db}")

        for freq in (25, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000):
            x = self._freq_to_x(freq, graph)
            painter.setPen(QPen(QColor("#3a414a"), 1))
            painter.drawLine(x, graph.top(), x, graph.bottom())
            if freq in (25, 100, 500, 1000, 5000, 10000, 20000):
                label = self._format_frequency(freq)
                metrics = painter.fontMetrics()
                label_width = metrics.horizontalAdvance(label)
                label_x = int(np.clip(x - label_width / 2, graph.left(), graph.right() - label_width))
                painter.setPen(QPen(QColor("#c6ccd3"), 1))
                painter.drawLine(x, graph.bottom() + 1, x, graph.bottom() + 5)
                painter.drawText(label_x, graph.bottom() + 23, label)

        painter.setPen(QPen(QColor("#c6ccd3"), 1))
        painter.drawText(6, graph.top() - 6, "dB")
        painter.drawText(graph.right() - 12, graph.bottom() + 37, "Hz")

        if len(self.bands_db) == 0:
            return

        trace_points = []
        for index, value in enumerate(self.bands_db):
            left = self._freq_to_x(self.edges[index], graph)
            right = self._freq_to_x(self.edges[index + 1], graph)
            width = max(1, right - left + 1)
            y = self._db_to_y(float(value), graph, min_db, max_db)
            painter.fillRect(left, y, width, graph.bottom() - y, QColor("#00f03c"))
            trace_value = float(self.average_bands_db[index])
            trace_y = self._db_to_y(trace_value, graph, min_db, max_db)
            trace_points.append(QPointF(left + max(1, width / 2), trace_y))

        selected_x = self._freq_to_x(selected_freq, graph)
        painter.setPen(QPen(QColor("#f2d16b"), 2))
        painter.drawLine(selected_x, graph.top(), selected_x, graph.bottom())

        painter.setPen(QPen(QColor("#ff2e78"), 2))
        for start, end in zip(trace_points, trace_points[1:]):
            painter.drawLine(start, end)
        painter.setBrush(QColor("#ff2e78"))
        for point in trace_points[::4]:
            painter.drawEllipse(point, 2, 2)

    def mouseMoveEvent(self, event):
        self._set_selected_from_x(event.position().x())
        self._show_selected_tooltip(event)

    def mousePressEvent(self, event):
        self._set_selected_from_x(event.position().x())
        self._show_selected_tooltip(event)

    def leaveEvent(self, event):
        QToolTip.hideText()
        super().leaveEvent(event)

    def _set_selected_from_x(self, x: float):
        graph = self._graph_rect()
        frequency = self._x_to_freq(x, graph)
        centers = np.sqrt(self.edges[:-1] * self.edges[1:])
        self.selected_index = int(np.argmin(np.abs(centers - frequency)))
        self.update()

    def _show_selected_tooltip(self, event):
        graph = self._graph_rect()
        if not graph.contains(event.position().toPoint()) or len(self.bands_db) == 0:
            QToolTip.hideText()
            return

        index = int(np.clip(self.selected_index, 0, len(self.bands_db) - 1))
        frequency = float(np.sqrt(self.edges[index] * self.edges[index + 1]))
        db_value = float(self.average_bands_db[index]) if len(self.average_bands_db) else -72.0
        text = f"{self._format_tooltip_frequency(frequency)}\n{db_value:.1f} dB avg"
        global_position = (
            event.globalPosition().toPoint()
            if hasattr(event, "globalPosition")
            else event.globalPos()
        )
        QToolTip.showText(global_position, text, self)

    def _graph_rect(self):
        return self.rect().adjusted(34, 18, -18, -42)

    def _db_to_y(self, value: float, graph, min_db: float, max_db: float) -> int:
        normalized = (float(np.clip(value, min_db, max_db)) - min_db) / (max_db - min_db)
        return int(graph.bottom() - normalized * graph.height())

    def _freq_to_x(self, frequency: float, graph) -> int:
        low = np.log10(25.0)
        high = np.log10(SAMPLE_RATE * 0.45)
        normalized = (np.log10(float(np.clip(frequency, 25.0, SAMPLE_RATE * 0.45))) - low) / (high - low)
        return int(graph.left() + normalized * graph.width())

    def _x_to_freq(self, x: float, graph) -> float:
        normalized = (float(x) - graph.left()) / max(1, graph.width())
        low = np.log10(25.0)
        high = np.log10(SAMPLE_RATE * 0.45)
        return 10.0 ** (low + np.clip(normalized, 0.0, 1.0) * (high - low))

    def _format_frequency(self, frequency: float) -> str:
        if frequency >= 1000.0:
            return f"{frequency / 1000.0:.1f}k"
        return f"{frequency:.0f}"

    def _format_tooltip_frequency(self, frequency: float) -> str:
        if frequency >= 1000.0:
            return f"{frequency / 1000.0:.2f} kHz"
        return f"{frequency:.1f} Hz"
