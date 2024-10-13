
import sys, os, time
import query_all

from PySide6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                               QTextEdit, QLineEdit, QLabel, QSlider, QCheckBox, QGroupBox,
                               QScrollArea, QProgressBar,
                               QGraphicsView, QGraphicsScene, QGraphicsPixmapItem)
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtCore import Qt

six_models = [
    "gpt-4o-mini",
    "claude-3-haiku-20240307",
    "gemini-1.5-flash-002",
    "gpt-4o",
    "claude-3-5-sonnet-20240620",
    "gemini-1.5-pro-002",
]

class TaskWidget(QWidget):
    def __init__(self, model_name):
        super().__init__()
        self.model_name = model_name
        layout = QVBoxLayout()

        label = QLabel(f"{model_name} Output:")
        layout.addWidget(label)

        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText(f"Enter input for {model_name}")
        layout.addWidget(self.input_field)

        self.output_area = QTextEdit()
        self.output_area.setReadOnly(True)
        layout.addWidget(self.output_area)

        self.setLayout(layout)

    def reset(self):
        self.input_field.clear()
        self.output_area.clear()

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Multi-LLM Query")
        self.setGeometry(100, 100, 1200, 900)

        self.main_layout = QVBoxLayout()
        self.setLayout(self.main_layout)

        # Add title at the top
        title_label = QLabel("Multi-LLM Query")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-size: 36px; font-weight: bold; padding: 5px;")
        title_label.setFixedHeight(40)  # Set a fixed height for the title
        self.main_layout.addWidget(title_label)

        # Reload button at the top
        self.reload_button = QPushButton("Reload")
        self.reload_button.clicked.connect(self.reload_tasks)
        self.main_layout.addWidget(self.reload_button)

        # Add temperature slider and model selector
        controls_layout = QHBoxLayout()

        # Temperature slider
        temp_layout = QVBoxLayout()
        temp_label = QLabel("Temperature:")
        temp_layout.addWidget(temp_label)

        slider_value_layout = QVBoxLayout()
        self.temp_slider = QSlider(Qt.Horizontal)
        self.temp_slider.setRange(0, 20)
        self.temp_slider.setValue(1)
        self.temp_value_label = QLabel("0.05", alignment=Qt.AlignCenter)
        self.temp_slider.valueChanged.connect(self.update_temp_label)

        slider_value_layout.addWidget(self.temp_slider)
        slider_value_layout.addWidget(self.temp_value_label)
        slider_value_layout.setSpacing(0)

        temp_layout.addLayout(slider_value_layout)

        # Model selector with scroll area
        model_group = QGroupBox("Models")
        model_layout = QVBoxLayout()
        self.model_checkboxes = []
        for model_name in six_models:
            checkbox = QCheckBox(model_name)
            self.model_checkboxes.append(checkbox)
            model_layout.addWidget(checkbox)

        scroll_area = QScrollArea()
        scroll_widget = QWidget()
        scroll_widget.setLayout(model_layout)
        scroll_area.setWidget(scroll_widget)
        scroll_area.setWidgetResizable(True)
        scroll_area.setFixedHeight(150)  # Set a fixed height for the scroll area

        model_group_layout = QVBoxLayout()
        model_group_layout.addWidget(scroll_area)
        model_group.setLayout(model_group_layout)

        controls_layout.addLayout(temp_layout, 1)
        controls_layout.addSpacing(20)
        controls_layout.addWidget(model_group, 1)

        # Set a fixed height for the controls section
        controls_widget = QWidget()
        controls_widget.setLayout(controls_layout)
        controls_widget.setFixedHeight(200)  # Adjust this value as needed

        self.main_layout.addWidget(controls_widget)

        # Add new buttons: Create Output Areas and Run
        button_layout = QHBoxLayout()
        self.create_output_areas_button = QPushButton("Create Input and Output Areas")
        self.create_output_areas_button.clicked.connect(self.create_output_areas)
        self.run_button = QPushButton("Run")
        self.run_button.clicked.connect(self.run_tasks)
        self.run_button.setEnabled(False)  # Disable Run button initially
        button_layout.addWidget(self.create_output_areas_button)
        button_layout.addWidget(self.run_button)
        self.main_layout.addLayout(button_layout)

        # Add image preview section
        image_layout = self.create_image_preview_section()
        self.main_layout.addLayout(image_layout)

        # Add prompt field
        self.prompt_field = QLineEdit()
        self.prompt_field.setPlaceholderText("Enter prompt for ALL models (OPTIONAL)")
        self.prompt_field.setStyleSheet("QLineEdit { color: green; }") ##  #4CAF50
        self.main_layout.addWidget(self.prompt_field)

        self.task_widgets = []
        self.task_layout = QVBoxLayout()

        # Add progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setAlignment(Qt.AlignCenter)  # Center-align the text
        self.progress_bar.setFormat("%p%")  # Show percentage
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid grey;
                border-radius: 5px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
                width: 10px;
            }
        """)
        self.main_layout.addWidget(self.progress_bar)

        self.main_layout.addLayout(self.task_layout)

        # Set initial focus to the prompt field
        self.prompt_field.setFocus()

    def update_temp_label(self, value):
        self.temp_value_label.setText(f"{value / 20:.2f}")

    def create_output_areas(self):
        # Clear existing task widgets
        for widget in self.task_widgets:
            widget.setParent(None)
        self.task_widgets.clear()

        # Remove existing layouts from task_layout
        while self.task_layout.count():
            item = self.task_layout.takeAt(0)
            if item.layout():
                while item.layout().count():
                    sub_item = item.layout().takeAt(0)
                    if sub_item.widget():
                        sub_item.widget().deleteLater()
            elif item.widget():
                item.widget().deleteLater()

        # Count selected models
        self.selected_model_names = [ checkbox.text() for checkbox in self.model_checkboxes 
                                          if checkbox.isChecked() ]
        nselected = len(self.selected_model_names)

        if nselected == 0:
            return

        # Calculate rows and columns
        cols = min(nselected, 3)
        rows = (nselected + cols - 1) // cols

        # Create new task widgets
        for model_name in self.selected_model_names:
            task_widget = TaskWidget(model_name)
            self.task_widgets.append(task_widget)

        # Add task widgets to layout
        for row in range(rows):
            row_layout = QHBoxLayout()
            for col in range(cols):
                index = row * cols + col
                if index < nselected:
                    row_layout.addWidget(self.task_widgets[index])
            self.task_layout.addLayout(row_layout)

        # Enable the Run button
        self.run_button.setEnabled(True)

    def update_progress(self, value):
        self.progress_bar.setValue(value)
        self.progress_bar.setFormat(f"{value}%")
        QApplication.processEvents()  # Ensure the UI updates

    def run_tasks(self):
        prompt = self.prompt_field.text()
        image_url = self.image_field.text()
        temp = float(self.temp_value_label.text())
        self.update_progress(0)
        total_tasks = len(self.task_widgets)

        for (idx,task_widget) in enumerate(self.task_widgets):
            if prompt:  # use the global prompt if it's not empty
                input_text = prompt
                task_widget.input_field.setText(prompt)
            else:  # use individual input field if global prompt empty
                input_text = task_widget.input_field.text()
            if not input_text:
                input_text = "hi"

            # Set the output
            user_content =  [ {"type": "text", "text": input_text} ]
            if image_url:
                user_content.append( {"type": "image_url", "image_url": {"url": image_url}} )
            messages = [
                {
                    "role": "user",
                    "content": user_content
                },
            ]
            model_name = task_widget.model_name
            llm_response = query_all.generate(model_name, messages, temperature=temp)
            task_widget.output_area.setText(llm_response)

            # Update progress bar
            progress = int((idx + 1) / total_tasks * 100)
            self.update_progress(progress)

            # Process events to update the UI
            QApplication.processEvents()

    def create_image_preview_section(self):
        """Creates the image preview section with file path field and preview area."""
        image_layout = QHBoxLayout()

        # Image File Path Field
        self.image_field = QLineEdit()
        self.image_field.setPlaceholderText("Enter image file path")  # Changed placeholder
        self.image_field.setStyleSheet("QLineEdit { color: green; }")
        self.image_field.textChanged.connect(self.load_image_preview)
        image_layout.addWidget(self.image_field)

        # Image Preview Area
        self.image_preview = QGraphicsView()
        self.image_scene = QGraphicsScene()
        self.image_preview.setScene(self.image_scene)
        self.image_preview.setFixedSize(200, 150)  # Set a fixed size for the preview
        image_layout.addWidget(self.image_preview)

        return image_layout

    def load_image_preview(self, file_path):
        """Loads and displays the image preview from a local file path."""
        self.image_scene.clear()
        try:
            # Use QImage directly for local files
            image = QImage(file_path) 

            if not image.isNull():
                pixmap = QPixmap.fromImage(image)
                pixmap = pixmap.scaled(self.image_preview.width(), self.image_preview.height(), Qt.KeepAspectRatio)
                item = QGraphicsPixmapItem(pixmap)
                self.image_scene.addItem(item)
            else:
                print(f"Error loading image: Invalid file or format: {file_path}") # More specific error message
        except Exception as e:
            print(f"Error displaying image: {e}")

    def reload_tasks(self):
        for task_widget in self.task_widgets:
            task_widget.setParent(None)
        self.task_widgets.clear()
        self.prompt_field.clear()
        self.run_button.setEnabled(False)
        self.update_progress(0)

        # Remove existing layouts from task_layout
        while self.task_layout.count():
            item = self.task_layout.takeAt(0)
            if item.layout():
                while item.layout().count():
                    sub_item = item.layout().takeAt(0)
                    if sub_item.widget():
                        sub_item.widget().deleteLater()
            elif item.widget():
                item.widget().deleteLater()

    def populate_fields(self, text):
        for task_widget in self.task_widgets:
            task_widget.input_field.setText(text)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
