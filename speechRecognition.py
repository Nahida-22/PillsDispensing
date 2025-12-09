import speech_recognition as sr

# Create recognizer instance
recognizer = sr.Recognizer()

print("Say 'hello' to test speech recognition...")

while True:
    try:
        with sr.Microphone() as mic:
            recognizer.adjust_for_ambient_noise(mic, duration=0.5)
            print("Listening...")
            audio = recognizer.listen(mic)

        # Convert speech to text
        text = recognizer.recognize_google(audio)
        text = text.lower()      # make it easier to compare

        print(f"You said: {text}")

        # Trigger condition
        if "hello" in text:
            print("👋 Dofbot detected! Speech recognition test successful!")
            break

    except sr.UnknownValueError:
        print("I couldn't understand that, please try again...")
    except sr.RequestError:
        print("Speech recognition service error. Check your internet connection.")
