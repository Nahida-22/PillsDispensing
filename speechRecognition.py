import speech_recognition as sr



# Create recognizer instance
recognizer = sr.Recognizer()

# Device index for your AUX headset
MIC_INDEX = 3

print("Say 'hello Dofbot' to test speech recognition...")

try:
    with sr.Microphone(device_index=MIC_INDEX) as mic:
        recognizer.adjust_for_ambient_noise(mic, duration=0.5)
        print("Listening...")

        while True:
            try:
                audio = recognizer.listen(mic, timeout=5, phrase_time_limit=5)
                text = recognizer.recognize_google(audio).lower()
                print(f"You said: {text}")

                if "hello" in text:
                    print("👋 Dofbot detected! Speech recognition test successful!")
                    break

            except sr.UnknownValueError:
                print("I couldn't understand that, please try again...")
            except sr.RequestError:
                print("Speech recognition service error. Check your internet connection.")
            except sr.WaitTimeoutError:
                print("No speech detected, listening again...")

except KeyboardInterrupt:
    print("\nStopped by user")
