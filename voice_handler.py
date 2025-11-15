import speech_recognition as sr
import pyttsx3
import threading

class VoiceHandler:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.tts_engine = pyttsx3.init()
        
        # Configure TTS
        self.tts_engine.setProperty('rate', 150)  # Speed
        self.tts_engine.setProperty('volume', 0.9)  # Volume
        
        # Try to get female voice if available
        voices = self.tts_engine.getProperty('voices')
        for voice in voices:
            if 'female' in voice.name.lower():
                self.tts_engine.setProperty('voice', voice.id)
                break
    
    def listen(self, timeout=5):
        """
        Listen for voice command
        Returns: recognized text or None
        """
        with sr.Microphone() as source:
            self.speak("Listening...")
            self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
            
            try:
                audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=5)
                text = self.recognizer.recognize_google(audio)
                return text.lower()
            except sr.WaitTimeoutError:
                return None
            except sr.UnknownValueError:
                self.speak("Sorry, I didn't understand that")
                return None
            except sr.RequestError:
                self.speak("Speech recognition service unavailable")
                return None
    
    def speak(self, text):
        """Speak text using TTS"""
        def _speak():
            self.tts_engine.say(text)
            self.tts_engine.runAndWait()
        
        # Run in separate thread to avoid blocking
        thread = threading.Thread(target=_speak)
        thread.start()
    
    def speak_blocking(self, text):
        """Speak text and wait for completion"""
        self.tts_engine.say(text)
        self.tts_engine.runAndWait()
