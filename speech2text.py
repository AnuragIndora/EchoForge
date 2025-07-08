import whisper
import os

class Speech2Text:
    def __init__(self, model):
        self.model = model

    def transcribe_audio(self, audio_file_path):
        try:
            if os.path.exists(audio_file_path):
                result = self.model.transcribe(audio_file_path)
                return result['text']
            else:
                print(f"Audio File Not Found: {audio_file_path}")
                return None
        except Exception as e:
            print(f"Error processing {audio_file_path}: {e}")
            return None

def main(input_folder, output_folder):
    # Load the Whisper model
    print("Model Loading Start...")
    model = whisper.load_model("medium")
    print("Model Loading Done!")

    speech_to_text = Speech2Text(model)

    # Process each .mp3 file in the input folder
    for file_name in os.listdir(input_folder):
        if file_name.endswith('.mp3'):
            audio_file_path = os.path.join(input_folder, file_name)
            print(f"Processing: {audio_file_path}")
            # Transcribe the audio file
            print(f"Transcription mode on...")
            transcription = speech_to_text.transcribe_audio(audio_file_path)
            print(f"Transcription done!")
            if transcription:
                # Create the output file name
                output_file_name = os.path.splitext(file_name)[0] + '.txt'
                output_file_path = os.path.join(output_folder, output_file_name)
                # Save the transcription to a .txt file
                with open(output_file_path, 'w') as file:
                    file.write(transcription)
                print(f"Transcription saved to: {output_file_path}")

if __name__ == "__main__":
    input_folder = "input" 
    output_folder = "output"  

    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    main(input_folder, output_folder)
