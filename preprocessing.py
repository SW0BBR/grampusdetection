import numpy as np
import pandas as pd
import pydub
import matplotlib.pyplot as plt
import os

def audit_to_csv(audit_path):
    """
    Converts the standard audit format into a csv file.
    Saves it in {audit_path}.csv
    """
    # Load audit file into python
    df = pd.read_csv(audit_path, delimiter = "\n", header=None)
    # Remove first 6 and last 2 rows, not relevant
    df = df[:-2]
    # Split into appropriate columns
    df = df[0].str.split("\t",0,expand=True)
    # Remove last column (contains none column)
    # df.drop(3, inplace=True, axis=1)
    # Rename indices
    df.columns = ["Time", "Duration", "Class"]
    # Make first element index 0
    df.reset_index(drop=True, inplace=True)
    # Set appropriate columns to floats instead of strings
    df["Time"] = df["Time"].astype(float)
    df["Duration"] = df["Duration"].astype(float)
    # Save as csv
    df.to_csv(r"{}".format(audit_path[:-4] + ".csv"), index = False)
    print("Converted {} to csv...".format(audit_path))
    return df

def save_voc(voc, clas):
    """
    Converts pydub AudioSegments into wav files.
    Saves the file under data/audio/{class}/{class}{number}.wav
    """
    # Desired result : data/audio/bp/bp0
    folderpath = "/media/alex/DOLPHINDRIV/data/audio/{}".format(clas)
    num = len(os.listdir(folderpath))
    namepath = "{}/{}_{}.wav".format(folderpath, clas, num)
    voc.export(namepath, format="wav")
    return namepath

def segment_vocs_mk2(begin_time, audit_dataframe, audiofile, save=True):
    """
    Creates mp3 vocalization files by slicing the original audio file and places them
    in the corresponding folder. Original dataframe gains a column in which the path to the call is noted.
    Takes in a dataframe made from audit file and begintime of that audiofile.
    """
    audit_dataframe["Vocalizations"] = None
    print(audit_dataframe.columns)

    # Classes that come with a specified duration
    dur_classes = ["buzz", "bp", "wbp", "whistle", "chirp", "ps", "cs"]
    # Classes that don't have a specified duration
    seq_classes = ["soc", "eoc", "s", "p"]
    # Variables related to clicking sequences
    soc_active = 0
    print("Extracting vocalizations ...")
    # Loop through audit csv, use occ_time and duration to cut out
    for row in audit_dataframe.itertuples():
        index, occ_time, dur, clas, _ = row
        sound_begin_index = (occ_time - begin_time) * 1000
        if clas in dur_classes:
            # Don't record soc if there's another class in it
            if soc_active != 0:
                soc_active = 0
            # Extract vocalization from audio file
            voc = audiofile[sound_begin_index: sound_begin_index + (dur * 1000)]
            # Either save the file and add the path to the dataframe
            if save == True:
                namepath = save_voc(voc,clas)
                audit_dataframe.at[index, "Vocalizations"] = namepath
            # Or add the actual wav file to the dataframe as np array
            else:
                voc_array = np.array(voc.get_array_of_samples())
                audit_dataframe.at[index, "Vocalizations"] = voc_array
        elif clas in seq_classes:
            # Duration of soc is time(eoc) - time(soc)
            if clas == "soc":
                soc_active = occ_time
            elif clas == "p" and soc_active != 0:
                # When soc contains pauses, save click section before pause,
                soc_begin_index = (soc_active - begin_time) * 1000
                click_duration = (occ_time - soc_active) * 1000
                voc = audiofile[soc_begin_index: soc_begin_index + click_duration]
                if save == True:
                    namepath = save_voc(voc, "eoc")
                    audit_dataframe.at[index, "Vocalizations"] = namepath
                # Move start of clicking to after the pause
                soc_active += dur
            elif clas == "eoc" and soc_active != 0:
                soc_begin_index = (soc_active - begin_time) * 1000
                dur_eoc = (occ_time - soc_active) * 1000
                voc = audiofile[soc_begin_index: soc_begin_index + dur_eoc]
                if save == True:
                    namepath = save_voc(voc,clas)
                    audit_dataframe.at[index, "Vocalizations"] = namepath
                else:
                    voc_array = np.array(voc.get_array_of_samples())
                    audit_dataframe.at[index, "Vocalizations"] = voc_array
                soc_active = 0
            # Surfacing has a set time of 1.1 seconds
            elif clas == "s":
                voc = audiofile[sound_begin_index: sound_begin_index + 1100]
                if save == True:
                    namepath = save_voc(voc,clas)
                    audit_dataframe.at[index, "Vocalizations"] = namepath
                else:
                    voc_array = np.array(voc.get_array_of_samples())
                    audit_dataframe.at[index, "Vocalizations"] = voc_array
    print("Extracted vocalizations!")
    return audit_dataframe

def extract_vocs_folder(audit_path, audio_folder_path):
    """
    Takes in a folder of audio samples. For each audio sample, cuts out vocalizations and saves them in
    data/{class}
    
    returns a list containing dataframes of each sample and its path to the mp3 of it.
    """
    started = False
    sample_dfs = []
    # Convert txt to dataframe, used to find time and duration
    audio_df_full = audit_to_csv(audit_path)
    audio_df_full["Vocalizations"] = None
    # tagon_time = float(audio_df_full[audio_df_full["Class"] == "tag on"]["Time"])
    tagon_time = 0
    # Variables for linking df and audio file
    audio_begin, audio_end = 0,0
    for audio_file in sorted(os.listdir(audio_folder_path)):
        # Load audio file
        print("Loading {} ...".format(audio_file))
        audio_file_path = os.path.join(audio_folder_path, audio_file)
        audio_file = pydub.AudioSegment.from_wav(audio_file_path)
        print("Loaded {} !".format(audio_file_path))
        # Calculate duration
        audio_file_duration = audio_file.duration_seconds
        audio_end += audio_file_duration
        # Check if file contains tagon, if not go to the next one
        if audio_end < tagon_time and started == False:
            audio_begin = audio_end
            sample_dfs.append(pd.DataFrame())
            started = True
            continue
        # Audiofile is synced with audio_df by making a df for just the audiofile
        audio_df_sample = audio_df_full[(audio_df_full["Time"] < audio_end) & (audio_df_full["Time"] > audio_begin)]
        audio_df_sample = segment_vocs_mk2(audio_begin, audio_df_sample, audio_file)
        # After segmenting audio files, update audio_begin to start the next audiofile at the correct time.
        sample_dfs.append(audio_df_sample)
        audio_begin = audio_end
    return sample_dfs

# test run, only run one audio in audio folder
path_to_audit = "../audio/gg13_238aaud.txt"
path_to_audio = "/media/alex/DOLPHINDRIV/gg238a"
voclist = extract_vocs_folder(path_to_audit, path_to_audio)