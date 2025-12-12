import speech_recognition as sr

print("Available microphone devices:\n")
for index, name in enumerate(sr.Microphone.list_microphone_names()):
    print(f"{index}: {name}")
