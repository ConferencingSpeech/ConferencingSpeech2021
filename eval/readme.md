# Track1
Run the track1 evaluate code:

    1. sh run_track1.sh
        wav_list: enhanced wav_list
        pathe: directory for enhanced wavs
        pathc: directory for clean wavs
        pathn: directory for noisy wavs
        result_list: path for track1 result file

# Track2
Run the track2 evaluate code:

    1. sh run_track2.sh
        tmp_path: path for temporary file
        enh_dir: directory for enhanced wavs
        ref_dir: directory for clean wavs
        mix_dir: directory for noisy wavs
        result_list: path for track2 result file

# Details
wav_list is the basename file in the following format:
    
    xxxx.wav
    yyyy.wav
    zzzz.wav

result_list is the objective score file in the following format:
    
    basename noisy_pesq enhance_pesq noisy_stoi enhance_stoi noisy_estoi enhance_estoi noisy_sisnr enhance_sisnr

