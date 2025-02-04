import tkinter as tk
from tkinter import filedialog, colorchooser
from PIL import Image, ImageTk, ImageGrab, ImageChops
import pyaudio
import wave
import numpy as np
import whisper
import torch
import polyllm
from kokoro.models import build_model
from kokoro.kokoro import generate as kokoro_generate

class AppConfig:
    """Configuration for application settings"""
    # Audio settings
    KOKORO_TTS_MODEL_PATH = 'kokoro/kokoro-v0_19.pth'
    KOKORO_VOICES_DIR = 'kokoro/voices'
    DEFAULT_VOICE_NAME = 'af_sky'
    KOKORO_SAMPLE_RATE = 24000
    WHISPER_MODEL_SIZE = "small"
    AUDIO_CHUNK_SIZE = 1024
    AUDIO_FORMAT = pyaudio.paInt16
    AUDIO_CHANNELS = 1
    AUDIO_USER_RATE = 16000
    TEMP_AUDIO_FILE_PATH = "temp_audio.wav"
    
    # LLM settings
    LLM_MODEL_NAME = "google/gemini-2.0-flash-exp"
    INITIAL_USER_MESSAGE = """
    Please respond with a one-sentence answer whenever possible.
    Here is an image that I may ask questions about.
    Ignore the image if it is not needed to respond to a question.
    """
    CANVAS_IMAGE_PATH = "current_canvas.png"

class AudioHandler:
    """Handles both TTS and STT functionality"""
    def __init__(self):
        self.config = AppConfig()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # TTS setup
        self.tts_model = build_model(self.config.KOKORO_TTS_MODEL_PATH, self.device)
        self.voice_pack = torch.load(f'{self.config.KOKORO_VOICES_DIR}/{self.config.DEFAULT_VOICE_NAME}.pt', 
                                   weights_only=True).to(self.device)
        # STT setup
        self.stt_model = whisper.load_model(self.config.WHISPER_MODEL_SIZE)
        # Audio recording setup
        self.audio_instance = pyaudio.PyAudio()
        self.stream = None
        self.frames = []
        self.is_recording = False

    def generate_speech(self, text):
        """Generate and play speech from text"""
        audio, _ = kokoro_generate(self.tts_model, text, self.voice_pack, 
                                 lang=self.config.DEFAULT_VOICE_NAME[0])
        self.play_audio(audio)

    def play_audio(self, audio_data):
        """Play audio data"""
        stream = self.audio_instance.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=self.config.KOKORO_SAMPLE_RATE,
            output=True
        )
        stream.write(audio_data.astype(np.float32).tobytes())
        stream.stop_stream()
        stream.close()

    def start_recording(self):
        """Start recording audio"""
        self.frames = []
        self.stream = self.audio_instance.open(
            format=self.config.AUDIO_FORMAT,
            channels=self.config.AUDIO_CHANNELS,
            rate=self.config.AUDIO_USER_RATE,
            input=True,
            frames_per_buffer=self.config.AUDIO_CHUNK_SIZE,
            stream_callback=self._recording_callback
        )
        self.stream.start_stream()
        self.is_recording = True

    def stop_recording(self):
        """Stop recording and save audio"""
        if self.is_recording:
            self.is_recording = False
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
            self._save_recording()
            return self.transcribe_audio()
        return ""

    def _recording_callback(self, in_data, frame_count, time_info, status):
        """Callback for audio recording"""
        if self.is_recording:
            audio_data = np.frombuffer(in_data, dtype=np.int16)
            # Calculate audio level for visual feedback
            audio_level = np.abs(audio_data).mean()
            self.frames.append(audio_data)
            # Print a simple visual indicator of audio level
            if audio_level > 1000:  # Adjust threshold as needed
                print("Recording: " + "█" * int(audio_level/1000))
        return (in_data, pyaudio.paContinue)

    def _save_recording(self):
        """Save recorded audio to file"""
        if self.frames:
            audio_data = np.concatenate(self.frames, axis=0)
            with wave.open(self.config.TEMP_AUDIO_FILE_PATH, "wb") as wf:
                wf.setnchannels(self.config.AUDIO_CHANNELS)
                wf.setsampwidth(self.audio_instance.get_sample_size(self.config.AUDIO_FORMAT))
                wf.setframerate(self.config.AUDIO_USER_RATE)
                wf.writeframes(audio_data.tobytes())

    def transcribe_audio(self):
        """Transcribe audio file to text"""
        try:
            result = self.stt_model.transcribe(self.config.TEMP_AUDIO_FILE_PATH, fp16=False)
            return result["text"]
        except Exception as e:
            print(f"Transcription error: {e}")
            return ""

class LLMHandler:
    """Handles communication with Language Model"""
    def __init__(self, config):
        self.config = config
        self.messages_to_llm = self._setup_initial_messages()

    def _setup_initial_messages(self):
        """Sets up initial message context for LLM"""
        return [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": self.config.INITIAL_USER_MESSAGE},
                    {"type": "image", "image": f"./{self.config.CANVAS_IMAGE_PATH}"},
                ],
            },
            {"role": "assistant", "content": "OK. I will use the image when relevant to query."},
            {"role": "user", "content": "DUMMY MSG TO BE REPLACED"},
        ]

    def query_llm(self, message_content):
        """Sends a query to the LLM and gets response"""
        print(f"LLM Query: {message_content}")
        self.messages_to_llm[-1] = {"role": "user", "content": message_content}
        try:
            response = polyllm.generate(
                model=self.config.LLM_MODEL_NAME,
                messages=self.messages_to_llm,
            )
            print(f"LLM Response: {response}")
            return response
        except Exception as e:
            error_message = f"LLM call failed: {e}"
            print(error_message)
            return error_message

class StudentAssistantApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Student Assistant")
        
        # Initialize handlers
        self.config = AppConfig()
        self.audio_handler = AudioHandler()
        self.llm_handler = LLMHandler(self.config)
        self.mic_active = False
        self.speaker_active = False
        
        # Initialize drawing variables
        self.drawing = False
        self.last_x = None
        self.last_y = None
        self.current_color = "#FFA500"
        
        # Undo/redo stacks
        self.undo_stack = []
        self.redo_stack = []
        
        # Top control panel
        control_frame = tk.Frame(root)
        control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # File operations buttons
        tk.Button(control_frame, text="Upload\nImage", command=self.upload_image, 
                 bg='white', fg='black').pack(side=tk.LEFT, padx=2)
        tk.Button(control_frame, text="Save\nCanvas", command=self.save_canvas,
                 bg='white', fg='black').pack(side=tk.LEFT, padx=2)
        tk.Button(control_frame, text="Clear", command=self.clear_canvas,
                 bg='white', fg='black').pack(side=tk.LEFT, padx=2)
        
        # Edit operations
        tk.Button(control_frame, text="Undo", command=self.undo,
                 bg='white', fg='black').pack(side=tk.LEFT, padx=2)
        tk.Button(control_frame, text="Redo", command=self.redo,
                 bg='white', fg='black').pack(side=tk.LEFT, padx=2)
        
        # Brush size control
        tk.Label(control_frame, text="Brush Size:").pack(side=tk.LEFT, padx=5)
        self.brush_size = tk.Entry(control_frame, width=5)
        self.brush_size.insert(0, "5")
        self.brush_size.pack(side=tk.LEFT)
        
        # Brush color
        tk.Label(control_frame, text="Brush Color:").pack(side=tk.LEFT, padx=5)
        self.color_display = tk.Canvas(control_frame, width=30, height=30, 
                                     bg=self.current_color, cursor='hand2')
        self.color_display.pack(side=tk.LEFT)
        self.color_display.create_rectangle(0, 0, 30, 30, outline='black')
        self.color_display.bind('<Button-1>', self.choose_color)
        
        # Main content area
        content_frame = tk.Frame(root)
        content_frame.pack(fill=tk.BOTH, expand=True, padx=5)
        
        # Left side with canvas
        left_frame = tk.Frame(content_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Canvas
        self.canvas = tk.Canvas(left_frame, bg='white', width=800, height=600)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind('<Button-1>', self.start_drawing)
        self.canvas.bind('<B1-Motion>', self.draw)
        self.canvas.bind('<ButtonRelease-1>', self.stop_drawing)
        
        # Right side controls
        right_frame = tk.Frame(content_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=5)
        
        # Frame for mic and speaker buttons
        button_frame = tk.Frame(right_frame)
        button_frame.pack(pady=5)
        
        # Mic and Speaker buttons side by side
        self.speaker_btn = tk.Button(button_frame, text="🔊\nSpeaker OFF",
                                   command=self.speaker_clicked,
                                   width=6, height=2, fg='red')
        self.speaker_btn.pack(side=tk.LEFT, padx=2)
        
        self.mic_btn = tk.Button(button_frame, text="🎤\nMic OFF", 
                                command=self.mic_clicked,
                                width=6, height=2, fg='red')
        self.mic_btn.pack(side=tk.LEFT, padx=2)
        
        # Text input area
        bottom_frame = tk.Frame(right_frame)
        bottom_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.text_input = tk.Entry(bottom_frame)
        self.text_input.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.text_input.bind('<Return>', self.send_message)
        
        send_btn = tk.Button(bottom_frame, text="SEND", command=self.send_message,
                            bg='white', fg='black')
        send_btn.pack(side=tk.RIGHT)
        
        # Message display area
        self.message_area = tk.Text(right_frame, height=10, width=60)
        self.message_area.pack(fill=tk.BOTH, expand=True)
        
        # Status bar
        self.status_bar = tk.Label(root, text="Ready", bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Bind keyboard shortcuts
        self.root.bind('<Control-z>', lambda e: self.undo())
        self.root.bind('<Control-y>', lambda e: self.redo())
        self.root.bind('<Control-s>', lambda e: self.save_canvas())
        self.root.bind('<space>', self.toggle_mic)

    def choose_color(self, event=None):
        color = colorchooser.askcolor(color=self.current_color)[1]
        if color:
            self.current_color = color
            self.color_display.configure(bg=color)

    def start_drawing(self, event):
        self.drawing = True
        self.last_x = event.x
        self.last_y = event.y

    def draw(self, event):
        if self.drawing:
            try:
                brush_size = int(self.brush_size.get())
            except ValueError:
                brush_size = 5
                self.brush_size.delete(0, tk.END)
                self.brush_size.insert(0, "5")

            self.canvas.create_line(self.last_x, self.last_y, event.x, event.y,
                                  width=brush_size, fill=self.current_color,
                                  capstyle=tk.ROUND, smooth=True)
            self.last_x = event.x
            self.last_y = event.y
            self.status_bar.config(text=f"Drawing at ({event.x}, {event.y})")

    def stop_drawing(self, event):
        if self.drawing:  # Only save state if we were actually drawing
            self.save_state()
        self.drawing = False
        
    def upload_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            try:
                image = Image.open(file_path)
                canvas_width = self.canvas.winfo_width()
                canvas_height = self.canvas.winfo_height()
                image.thumbnail((canvas_width, canvas_height))
                self.photo = ImageTk.PhotoImage(image)
                self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
            except Exception as e:
                print(f"Error loading image: {e}")
    
    def save_canvas(self, event=None):
        """Save the canvas content as a PNG file"""
        file_path = filedialog.asksaveasfilename(defaultextension=".png",
                                                filetypes=[("PNG files", "*.png")])
        if file_path:
            self._save_canvas_to_file(file_path)
            self.status_bar.config(text=f"Canvas saved to {file_path}")

    def _save_canvas_to_file(self, file_path):
        """Helper to save canvas content to a specific file"""
        # Get canvas dimensions
        x = self.canvas.winfo_rootx()
        y = self.canvas.winfo_rooty()
        width = self.canvas.winfo_width()
        height = self.canvas.winfo_height()
        
        # Create a new image with white background
        img = Image.new('RGB', (width, height), 'white')
        
        # Update canvas and capture the area
        self.canvas.update()
        screenshot = ImageGrab.grab(bbox=(x, y, x+width, y+height))
        
        # Paste screenshot onto white background and save
        img.paste(screenshot)
        img.save(file_path)
            
    def save_state(self):
        """Save current canvas state for undo/redo"""
        # Save all canvas items as a state
        state = []
        for item in self.canvas.find_all():
            item_type = self.canvas.type(item)
            coords = self.canvas.coords(item)
            
            if item_type == "line":
                config = {
                    'type': 'line',
                    'fill': self.canvas.itemcget(item, 'fill'),
                    'width': self.canvas.itemcget(item, 'width'),
                    'capstyle': self.canvas.itemcget(item, 'capstyle')
                }
            elif item_type == "image":
                config = {
                    'type': 'image',
                    'image': self.canvas.itemcget(item, 'image')
                }
            state.append((coords, config))
            
        # Only add state if it's different from the last state
        if not self.undo_stack or state != self.undo_stack[-1]:
            self.undo_stack.append(state)
            self.redo_stack.clear()  # Clear redo stack when new action is performed
        
    def undo(self, event=None):
        """Undo last drawing action"""
        if len(self.undo_stack) > 1:  # Need at least 2 states to undo
            current_state = self.undo_stack.pop()
            self.redo_stack.append(current_state)
            self.canvas.delete("all")
            # Restore previous state
            prev_state = self.undo_stack[-1]
            for coords, config in prev_state:
                if config['type'] == 'line':
                    line_config = {k: v for k, v in config.items() if k != 'type'}
                    self.canvas.create_line(*coords, **line_config)
                elif config['type'] == 'image':
                    self.canvas.create_image(*coords, image=self.photo)
            self.status_bar.config(text="Undo performed")
            
    def redo(self, event=None):
        """Redo last undone action"""
        if self.redo_stack:
            state = self.redo_stack.pop()
            self.undo_stack.append(state)
            self.canvas.delete("all")
            # Restore the state
            for coords, config in state:
                if config['type'] == 'line':
                    line_config = {k: v for k, v in config.items() if k != 'type'}
                    self.canvas.create_line(*coords, **line_config)
                elif config['type'] == 'image':
                    self.canvas.create_image(*coords, image=self.photo)
            self.status_bar.config(text="Redo performed")
    
    def clear_canvas(self):
        self.save_state()  # Save current state before clearing
        self.canvas.delete("all")
        self.status_bar.config(text="Canvas cleared")
    
    def toggle_mic(self, event=None):
        """Toggle microphone state when space bar is pressed"""
        self.mic_clicked()
        
    def mic_clicked(self):
        if not self.mic_active:
            self.mic_active = True
            self.mic_btn.configure(text="🎤\nMic ON", fg='green')
            self.audio_handler.start_recording()
            self.message_area.insert(tk.END, "Recording started... (Speak now)\n")
            self.message_area.see(tk.END)
            self.status_bar.config(text="Recording in progress...")
        else:
            self.mic_active = False
            self.mic_btn.configure(text="🎤\nMic OFF", fg='red')
            self.status_bar.config(text="Processing recording...")
            transcribed_text = self.audio_handler.stop_recording()
            
            if transcribed_text.strip():
                self.message_area.insert(tk.END, f"You said: {transcribed_text}\n")
                self.message_area.see(tk.END)
                self.status_bar.config(text="Querying LLM...")
                
                # Save canvas state and query LLM
                self._save_canvas_to_file(self.config.CANVAS_IMAGE_PATH)
                llm_response = self.llm_handler.query_llm(transcribed_text)
                self.message_area.insert(tk.END, f"Assistant: {llm_response}\n")
                self.message_area.see(tk.END)
                
                # Generate speech if enabled
                if self.speaker_active:
                    self.status_bar.config(text="Generating speech...")
                    self.audio_handler.generate_speech(llm_response)
                
                self.status_bar.config(text="Ready")
            else:
                self.message_area.insert(tk.END, "No speech detected. Please try again.\n")
                self.message_area.see(tk.END)
                self.status_bar.config(text="Ready")
    
    def speaker_clicked(self):
        self.speaker_active = not self.speaker_active
        if self.speaker_active:
            self.speaker_btn.configure(text="🔊\nSpeaker ON", fg='green')
            self.message_area.insert(tk.END, "Speaker enabled\n")
        else:
            self.speaker_btn.configure(text="🔊\nSpeaker OFF", fg='red')
            self.message_area.insert(tk.END, "Speaker disabled\n")
    
    def send_message(self, event=None):
        message = self.text_input.get()
        if message:
            self.message_area.insert(tk.END, f"You: {message}\n")
            # Save canvas state and query LLM
            self._save_canvas_to_file(self.config.CANVAS_IMAGE_PATH)
            llm_response = self.llm_handler.query_llm(message)
            self.message_area.insert(tk.END, f"Assistant: {llm_response}\n")
            # Generate speech if enabled
            if self.speaker_active:
                self.audio_handler.generate_speech(llm_response)
            self.text_input.delete(0, tk.END)

if __name__ == "__main__":
    root = tk.Tk()
    app = StudentAssistantApp(root)
    root.mainloop()
