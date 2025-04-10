"""
NEXUS Floating Assistant UI

This module implements a floating, always-on-top UI for the NEXUS system
that can display information, receive commands, and show real-time feedback
while NEXUS interacts with other applications.
"""
import os
import sys
import time
import logging
import threading
import tkinter as tk
from tkinter import ttk, scrolledtext
from typing import Dict, List, Callable, Optional, Tuple, Any
import queue
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

# Import NEXUS components
from src.ai_core.automation.safety_manager import SafetyManager

class FloatingAssistantUI:
    """
    Floating, always-on-top UI for NEXUS
    
    This window stays visible even when other applications are in focus,
    allowing NEXUS to display information and receive commands while 
    performing tasks across different applications.
    """
    
    def __init__(self, config: Dict = None):
        """Initialize the floating UI with configuration"""
        self.config = config or {}
        
        # Initialize message queues
        self.output_queue = queue.Queue()  # For messages to display
        self.input_queue = queue.Queue()   # For user input
        self.command_queue = queue.Queue() # For commands to execute
        
        # Track UI state
        self.is_running = False
        self.is_minimized = False
        self.awaiting_input = False
        self.current_query = ""
        
        # Message history
        self.message_history = []
        self.max_history = self.config.get("max_history", 100)
        
        # UI components
        self.root = None
        self.input_field = None
        self.output_area = None
        self.status_bar = None
        self.submit_button = None
        
        # Callback registry
        self.callbacks = {
            "on_input": [],
            "on_command": [],
            "on_close": []
        }
        
        # UI configurations
        self.theme = self.config.get("theme", "dark")
        self.opacity = self.config.get("opacity", 0.9)  # 0.0 to 1.0
        self.width = self.config.get("width", 400)
        self.height = self.config.get("height", 600)
        
        logger.info("Floating Assistant UI initialized")
    
    def start(self, position: Tuple[int, int] = None):
        """Start the floating UI in a separate thread"""
        if self.is_running:
            logger.warning("UI is already running")
            return
            
        # Start UI thread
        self.ui_thread = threading.Thread(target=self._run_ui, args=(position,))
        self.ui_thread.daemon = True
        self.ui_thread.start()
        
        self.is_running = True
        logger.info("Floating Assistant UI started")
    
    def _run_ui(self, position: Tuple[int, int] = None):
        """Run the UI in a thread"""
        self.root = tk.Tk()
        self.root.title("NEXUS Assistant")
        
        # Set window size
        self.root.geometry(f"{self.width}x{self.height}")
        
        # Make window stay on top
        self.root.attributes("-topmost", True)
        
        # Set window position if specified
        if position:
            self.root.geometry(f"+{position[0]}+{position[1]}")
        
        # Set window transparency
        try:
            # Windows specific
            self.root.attributes("-alpha", self.opacity)
        except:
            logger.warning("Transparency not supported on this platform")
            
        # Configure the UI theme
        self._setup_theme()
        
        # Build UI components
        self._build_ui()
        
        # Set up update loops
        self.root.after(100, self._process_output_queue)
        
        # Handle window close
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        
        # Start main loop
        self.root.mainloop()
        
        # When loop exits, set running to False
        self.is_running = False
    
    def _setup_theme(self):
        """Configure the UI theme colors and styles"""
        if self.theme == "dark":
            bg_color = "#1E1E1E"
            fg_color = "#FFFFFF"
            input_bg = "#2D2D2D"
            button_bg = "#0078D7"
            button_fg = "#FFFFFF"
            
            style = ttk.Style()
            style.theme_use('clam')  # Use a theme that we can modify
            
            style.configure("TFrame", background=bg_color)
            style.configure("TLabel", background=bg_color, foreground=fg_color)
            style.configure("TButton", background=button_bg, foreground=button_fg)
            
            self.root.configure(bg=bg_color)
            
        elif self.theme == "light":
            bg_color = "#F0F0F0"
            fg_color = "#000000"
            input_bg = "#FFFFFF"
            button_bg = "#0078D7"
            button_fg = "#FFFFFF"
            
            style = ttk.Style()
            style.theme_use('clam')
            
            style.configure("TFrame", background=bg_color)
            style.configure("TLabel", background=bg_color, foreground=fg_color)
            style.configure("TButton", background=button_bg, foreground=button_fg)
            
            self.root.configure(bg=bg_color)
        
        # Store theme colors for later use
        self.colors = {
            "bg": bg_color,
            "fg": fg_color,
            "input_bg": input_bg,
            "button_bg": button_bg,
            "button_fg": button_fg,
            "nexus_msg": "#0078D7",  # NEXUS message color
            "user_msg": "#22B14C",   # User message color
            "error_msg": "#FF0000",  # Error message color
            "warning_msg": "#FFA500"  # Warning message color
        }
    
    def _build_ui(self):
        """Build all UI components"""
        # Create main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create title label
        title_label = ttk.Label(main_frame, text="NEXUS Adaptive Assistant", 
                               font=("Arial", 14, "bold"))
        title_label.pack(pady=(0, 10))
        
        # Create output text area
        self.output_area = scrolledtext.ScrolledText(main_frame, height=15, 
                                                   bg=self.colors["input_bg"], 
                                                   fg=self.colors["fg"],
                                                   wrap=tk.WORD)
        self.output_area.pack(fill=tk.BOTH, expand=True, pady=5)
        self.output_area.config(state=tk.DISABLED)  # Read-only
        
        # Create input field
        input_frame = ttk.Frame(main_frame)
        input_frame.pack(fill=tk.X, pady=5)
        
        self.input_field = ttk.Entry(input_frame)
        self.input_field.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.input_field.bind("<Return>", self._on_submit)
        
        self.submit_button = ttk.Button(input_frame, text="Send", 
                                      command=self._on_submit)
        self.submit_button.pack(side=tk.RIGHT, padx=(5, 0))
        
        # Create status bar
        self.status_bar = ttk.Label(main_frame, text="Ready", anchor=tk.W)
        self.status_bar.pack(fill=tk.X, pady=(5, 0))
        
        # Create control buttons
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=5)
        
        clear_button = ttk.Button(control_frame, text="Clear", 
                                command=self.clear_output)
        clear_button.pack(side=tk.LEFT, padx=2)
        
        minimize_button = ttk.Button(control_frame, text="Minimize", 
                                   command=self._toggle_minimize)
        minimize_button.pack(side=tk.LEFT, padx=2)
        
        settings_button = ttk.Button(control_frame, text="Settings", 
                                   command=self._show_settings)
        settings_button.pack(side=tk.LEFT, padx=2)
        
        help_button = ttk.Button(control_frame, text="Help", 
                               command=self._show_help)
        help_button.pack(side=tk.LEFT, padx=2)
        
        # Display welcome message
        self.show_message("Welcome to NEXUS Adaptive Assistant", "system")
        self.show_message("I can help you with screen automation and learning tasks.", "nexus")
        self.show_message("Type a command or ask for help.", "nexus")
        
        # Focus on input field
        self.input_field.focus()
    
    def _process_output_queue(self):
        """Process messages in the output queue"""
        try:
            while True:
                # Get message from queue (non-blocking)
                message = self.output_queue.get_nowait()
                
                # Display message
                self.show_message(message["text"], message["type"])
                
                # Mark task as done
                self.output_queue.task_done()
                
        except queue.Empty:
            # Queue is empty, schedule next check
            pass
        finally:
            # Schedule next check
            if self.is_running and self.root:
                self.root.after(100, self._process_output_queue)
    
    def _on_submit(self, event=None):
        """Handle submit button click or Enter key"""
        # Get text from input field
        text = self.input_field.get().strip()
        
        # Clear input field
        self.input_field.delete(0, tk.END)
        
        if not text:
            return
            
        # Display user input
        self.show_message(text, "user")
        
        # Add to input queue
        self.input_queue.put(text)
        
        # Call registered callbacks
        for callback in self.callbacks["on_input"]:
            try:
                callback(text)
            except Exception as e:
                logger.error(f"Error in input callback: {e}")
        
        # If awaiting input for a specific query, this satisfies it
        if self.awaiting_input:
            self.awaiting_input = False
    
    def _on_close(self):
        """Handle window close"""
        # Call registered callbacks
        for callback in self.callbacks["on_close"]:
            try:
                callback()
            except Exception as e:
                logger.error(f"Error in close callback: {e}")
        
        self.is_running = False
        self.root.destroy()
    
    def _toggle_minimize(self):
        """Toggle between minimized and normal state"""
        if self.is_minimized:
            # Restore window
            self.root.geometry(f"{self.width}x{self.height}")
            self.is_minimized = False
        else:
            # Minimize to just title bar and input
            minimized_height = 100
            self.root.geometry(f"{self.width}x{minimized_height}")
            self.is_minimized = True
    
    def _show_settings(self):
        """Show settings dialog"""
        settings_window = tk.Toplevel(self.root)
        settings_window.title("NEXUS Settings")
        settings_window.attributes("-topmost", True)
        
        # Center the window
        settings_window.geometry("400x300")
        
        # Create settings UI
        settings_frame = ttk.Frame(settings_window, padding=10)
        settings_frame.pack(fill=tk.BOTH, expand=True)
        
        # Theme selection
        ttk.Label(settings_frame, text="Theme:").grid(row=0, column=0, sticky=tk.W, pady=5)
        theme_var = tk.StringVar(value=self.theme)
        theme_dropdown = ttk.Combobox(settings_frame, textvariable=theme_var)
        theme_dropdown['values'] = ('dark', 'light')
        theme_dropdown.grid(row=0, column=1, sticky=tk.W, pady=5)
        
        # Opacity slider
        ttk.Label(settings_frame, text="Opacity:").grid(row=1, column=0, sticky=tk.W, pady=5)
        opacity_var = tk.DoubleVar(value=self.opacity)
        opacity_slider = ttk.Scale(settings_frame, from_=0.1, to=1.0, 
                                 variable=opacity_var, orient=tk.HORIZONTAL)
        opacity_slider.grid(row=1, column=1, sticky=tk.W+tk.E, pady=5)
        
        # Size controls
        ttk.Label(settings_frame, text="Width:").grid(row=2, column=0, sticky=tk.W, pady=5)
        width_var = tk.IntVar(value=self.width)
        width_entry = ttk.Entry(settings_frame, textvariable=width_var, width=10)
        width_entry.grid(row=2, column=1, sticky=tk.W, pady=5)
        
        ttk.Label(settings_frame, text="Height:").grid(row=3, column=0, sticky=tk.W, pady=5)
        height_var = tk.IntVar(value=self.height)
        height_entry = ttk.Entry(settings_frame, textvariable=height_var, width=10)
        height_entry.grid(row=3, column=1, sticky=tk.W, pady=5)
        
        # Save button
        def save_settings():
            # Update settings
            self.theme = theme_var.get()
            self.opacity = opacity_var.get()
            self.width = width_var.get()
            self.height = height_var.get()
            
            # Apply settings
            try:
                self.root.attributes("-alpha", self.opacity)
            except:
                pass
                
            self.root.geometry(f"{self.width}x{self.height}")
            
            # Recreate UI with new theme
            for widget in self.root.winfo_children():
                widget.destroy()
            
            self._setup_theme()
            self._build_ui()
            
            # Close settings window
            settings_window.destroy()
        
        save_button = ttk.Button(settings_frame, text="Save", command=save_settings)
        save_button.grid(row=4, column=0, columnspan=2, pady=10)
    
    def _show_help(self):
        """Show help dialog"""
        help_window = tk.Toplevel(self.root)
        help_window.title("NEXUS Help")
        help_window.attributes("-topmost", True)
        
        help_window.geometry("500x400")
        
        help_text = """
NEXUS Adaptive Assistant Help

Commands:
- help: Show this help message
- clear: Clear the output area
- exit/quit: Close the assistant
- minimize: Minimize the window
- settings: Show settings dialog

Automation Commands:
- automate [app]: Start automating an application
- learn [action]: Learn a new automation pattern
- recall [pattern]: Recall and execute a learned pattern
- stop: Stop current automation

Examples:
- automate chrome
- learn login sequence
- recall weekly report
- stop

For more help, ask a question or describe what you want to do.
        """
        
        help_frame = ttk.Frame(help_window, padding=10)
        help_frame.pack(fill=tk.BOTH, expand=True)
        
        help_area = scrolledtext.ScrolledText(help_frame, wrap=tk.WORD)
        help_area.pack(fill=tk.BOTH, expand=True)
        help_area.insert(tk.END, help_text)
        help_area.config(state=tk.DISABLED)
    
    def show_message(self, message: str, msg_type: str = "nexus"):
        """Display a message in the output area"""
        if not self.root:
            # UI not initialized yet, add to queue
            self.output_queue.put({"text": message, "type": msg_type})
            return
            
        # Ensure UI updates happen in the main thread
        def _update_ui():
            # Enable editing
            self.output_area.config(state=tk.NORMAL)
            
            # Add timestamp
            timestamp = datetime.now().strftime("%H:%M:%S")
            
            # Set tag based on message type
            tag = msg_type.lower()
            
            # Add prefix based on message type
            if msg_type == "user":
                prefix = f"[{timestamp}] You: "
                tag_color = self.colors["user_msg"]
            elif msg_type == "nexus":
                prefix = f"[{timestamp}] NEXUS: "
                tag_color = self.colors["nexus_msg"]
            elif msg_type == "error":
                prefix = f"[{timestamp}] ERROR: "
                tag_color = self.colors["error_msg"]
            elif msg_type == "warning":
                prefix = f"[{timestamp}] WARNING: "
                tag_color = self.colors["warning_msg"]
            elif msg_type == "system":
                prefix = f"[{timestamp}] SYSTEM: "
                tag_color = self.colors["fg"]
            else:
                prefix = f"[{timestamp}] {msg_type.upper()}: "
                tag_color = self.colors["fg"]
            
            # Configure tag
            self.output_area.tag_configure(tag, foreground=tag_color)
            
            # Insert with newline if not first message
            if self.output_area.index('end-1c') != '1.0':
                self.output_area.insert(tk.END, "\n")
                
            # Insert prefix and message
            self.output_area.insert(tk.END, prefix, tag)
            self.output_area.insert(tk.END, message)
            
            # Scroll to bottom
            self.output_area.see(tk.END)
            
            # Disable editing
            self.output_area.config(state=tk.DISABLED)
            
            # Add to message history
            self.message_history.append({
                "timestamp": timestamp,
                "type": msg_type,
                "text": message
            })
            
            # Trim history if needed
            if len(self.message_history) > self.max_history:
                self.message_history = self.message_history[-self.max_history:]
        
        # If called from another thread, schedule update
        if threading.current_thread() is not threading.main_thread():
            if self.root:
                self.root.after(0, _update_ui)
            else:
                self.output_queue.put({"text": message, "type": msg_type})
        else:
            _update_ui()
    
    def clear_output(self):
        """Clear the output area"""
        if not self.root:
            return
            
        # Enable editing
        self.output_area.config(state=tk.NORMAL)
        
        # Clear all text
        self.output_area.delete(1.0, tk.END)
        
        # Disable editing
        self.output_area.config(state=tk.DISABLED)
        
        # Show system message
        self.show_message("Output cleared", "system")
    
    def update_status(self, status: str):
        """Update the status bar text"""
        if not self.root or not self.status_bar:
            return
            
        # If called from another thread, schedule update
        if threading.current_thread() is not threading.main_thread():
            if self.root:
                self.root.after(0, lambda: self.status_bar.config(text=status))
        else:
            self.status_bar.config(text=status)
    
    def ask_question(self, question: str, options: List[str] = None) -> str:
        """
        Ask user a question and wait for response
        
        This method is blocking and returns the user's response.
        If options are provided, they will be displayed as buttons.
        """
        if not self.is_running:
            logger.error("UI is not running")
            return ""
            
        # Create an event to wait for response
        response_event = threading.Event()
        response = {"text": ""}
        
        def _handle_ui_question():
            # Show the question
            self.show_message(question, "nexus")
            
            # If options provided, create buttons
            if options:
                # Create button frame
                button_frame = ttk.Frame(self.root)
                button_frame.pack(fill=tk.X, pady=5)
                
                def option_selected(option):
                    # Remove buttons
                    button_frame.destroy()
                    
                    # Show selected option
                    self.show_message(option, "user")
                    
                    # Set response
                    response["text"] = option
                    
                    # Signal that response is ready
                    response_event.set()
                
                # Create buttons for each option
                for option in options:
                    btn = ttk.Button(button_frame, text=option, 
                                   command=lambda o=option: option_selected(o))
                    btn.pack(side=tk.LEFT, padx=2)
            else:
                # Set flag to capture next input
                self.awaiting_input = True
                self.current_query = question
                
                # Register temporary input handler
                def handle_input(text):
                    if self.current_query == question:
                        response["text"] = text
                        response_event.set()
                
                handler_id = self.register_callback("on_input", handle_input)
        
        # Schedule UI update in main thread
        if self.root:
            self.root.after(0, _handle_ui_question)
        
        # Wait for response with timeout
        response_event.wait(timeout=300)  # 5 minute timeout
        
        # If options weren't used, clean up callback
        if not options:
            self.awaiting_input = False
            self.current_query = ""
        
        return response["text"]
    
    def register_callback(self, event_type: str, callback: Callable) -> int:
        """Register a callback for an event"""
        if event_type not in self.callbacks:
            raise ValueError(f"Unknown event type: {event_type}")
            
        # Add callback to list
        self.callbacks[event_type].append(callback)
        
        # Return index for unregistering
        return len(self.callbacks[event_type]) - 1
    
    def unregister_callback(self, event_type: str, callback_id: int):
        """Unregister a callback"""
        if event_type not in self.callbacks:
            raise ValueError(f"Unknown event type: {event_type}")
            
        if callback_id < 0 or callback_id >= len(self.callbacks[event_type]):
            raise ValueError(f"Invalid callback ID: {callback_id}")
            
        # Remove callback
        self.callbacks[event_type].pop(callback_id)
    
    def stop(self):
        """Stop the UI"""
        if not self.is_running:
            return
            
        # Signal thread to stop
        if self.root:
            self.root.after(0, self.root.destroy)
            
        self.is_running = False
        
        # Wait for thread to finish
        if hasattr(self, 'ui_thread') and self.ui_thread.is_alive():
            self.ui_thread.join(timeout=1.0)

# Standalone test function
def test_floating_ui():
    """Test the floating UI"""
    ui = FloatingAssistantUI(config={
        "theme": "dark",
        "opacity": 0.9,
        "width": 400,
        "height": 600
    })
    
    # Start the UI
    ui.start()
    
    # Wait for UI to initialize
    time.sleep(1)
    
    # Show some messages
    ui.show_message("This is a test message", "nexus")
    ui.show_message("This is a warning message", "warning")
    ui.show_message("This is an error message", "error")
    
    # Test input
    def on_input(text):
        if text.lower() == "exit" or text.lower() == "quit":
            ui.stop()
        else:
            ui.show_message(f"You typed: {text}", "nexus")
    
    # Register callback
    ui.register_callback("on_input", on_input)
    
    try:
        # Keep main thread alive
        while ui.is_running:
            time.sleep(0.1)
    except KeyboardInterrupt:
        # Stop on Ctrl+C
        ui.stop()

if __name__ == "__main__":
    test_floating_ui()
