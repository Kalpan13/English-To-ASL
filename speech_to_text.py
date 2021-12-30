import speech_recognition as sr
mic = sr.Microphone()
r = sr.Recognizer()
r.dynamic_energy_threshold = False

def convert_to_text():
    with mic as source:
        r.adjust_for_ambient_noise(source)
        audio = r.listen(source, timeout = 7)
        text = r.recognize_google(audio)
        print(text)
        return text