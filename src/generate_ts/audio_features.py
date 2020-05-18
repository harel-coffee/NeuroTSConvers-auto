from pydub import AudioSegment
import os, glob, argparse
import myprosody as mysp
import wave
import librosa
import scipy

def split_audio (audio):
    # Clear temporary audio files if exist
    os.system ("rm spplited_audio/*")

    t1 = 0 * 1000 #Works in milliseconds
    t2 = 0.6025 * 1000

    shank = audio[t1:t2]
    shank.export('spplited_audio/0.wav', format="wav")
    combin = shank

    start = 0.6025
    for i in range (1,50):
        t1 = start * 1000
        t2 = (start + 1.205) * 1000
        shank = audio[t1:t2]
        shank. export('spplited_audio/%d.wav'%i, format="wav")
        start += 1.205
        combin += shank

    combin. export('spplited_audio/combin.wav', format="wav")
    shanks_files = glob.glob ("spplited_audio/*.wav")

    return (shanks_files)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", help="the path of the file to process.")
    parser.add_argument("out_dir", help="the path where to store the results.")
    parser.add_argument("--language", "-lg", default = "fr", choices = ["fr", "eng"], help="Language.")
    parser.add_argument("--left", "-l", help="Process participant speech.", action="store_true")
    parser.add_argument("--outname", "-n", help="Rename output file names by default.", action="store_true")

    args = parser.parse_args()

    data_dir = args. data_dir
    out_dir = args. out_dir

    if args. out_dir[-1] != '/':
    	args. out_dir += '/'

    # create output dir file if does not exist
    if not os.path.exists (args. out_dir):
    	os.makedirs (args. out_dir)

    conversation_name = data_dir.split ('/')[-1]

    if conversation_name == "" or args.outname:
    	if args. left:
    		conversation_name = "audio_features_left"
    	else:
    		conversation_name = "audio_features"


    output_filename_pkl = out_dir +  conversation_name + ".pkl"

    print ("---------", conversation_name, "---------")
    # Index variable
    physio_index = [0.6025]
    for i in range (1, 50):
    	physio_index. append (1.205 + physio_index [i - 1])

    # Read audio, and transcription file
    for file in glob.glob(data_dir + "/*.wav"):
        if args.left and "left" in file:
            audio_file = file
            break
        elif "right" in file:
            audio_file = file
            break

    y, s = librosa.load (audio_file, sr=48000)

    #librosa.output.write_wav('spplited_audio/librosa.wav', y, s)
    scipy.io.wavfile.write('spplited_audio/librosa.wav', rate = 48000, data = y)

    y, s = librosa.load ('spplited_audio/librosa.wav', sr=None)
    print (s)

    '''wave_file = wave.open (audio_file)
    frame_rate = wave_file.getframerate()
    print (frame_rate)'''

    #audio = AudioSegment.from_wav(audio_file)
    #files = split_audio (audio)

    #for i in range (50):
    shank_file = 'spplited_audio/%d.wav'%i
    p = "sun"
    c = '/home/youssef/Github/PhysSocial/spplited_audio/'
    print (mysp.myspsr(p, c))


    #os.system ("rm spplited_audio/*")
