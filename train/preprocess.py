import json
import pandas as pd
import os
import shutil


def get_metadata(file):
    
    """ Creates dataframe from the json file containing video info and labels.
    
        Args:
            file: filename and path to the json.

        Return:
            A formatted pandas dataframe of the json file.
    """
    
    # read and load json file from specified path
    with open(file, 'r') as data_file:
        instance_json = json.load(data_file)

    # convert json format to readable pandas df
    metadata = pd.DataFrame()
    for i in range(0, len(instance_json)):

        label_ins = instance_json[i]
        gloss = label_ins['gloss']

        for i in range(0, len(label_ins['instances'])):

            label_ins['instances'][i]['gloss'] = gloss
            frame = pd.Series(label_ins['instances'][i]).to_frame().T
            metadata = pd.concat([metadata, frame])

    return metadata



def create_gloss_variations(df):
    # combine gloss and variation id columns
    df['gloss'] = df['gloss'] + df['variation_id'].astype(str)
    return df



def convert_sample_type(df, og_split = 'test', tgt_split = 'train'):
    # reassign test samples to train
    df['split'] = df['split'].replace({og_split: tgt_split})
    return df


def videos_to_folders(df, to_folder='data', type='copy', prototype=True):

    if prototype:
        df = df[df['gloss'].isin(['hello0', 'love0', 'thank you0'])]

    paths = df[['split', 'gloss']].drop_duplicates()#.apply(lambda row: os.path.join(*row), axis=1)
    for s, g in zip(paths['split'], paths['gloss']):
        move_to = os.path.join(to_folder, s, g)
        if not os.path.exists(move_to):
            os.makedirs(move_to)

        vid_ids = df[(df['split'] == s) & (df['gloss'] == g)]['video_id']
        for id in vid_ids:
            vid = os.path.join('videos', str(id)) + '.mp4'
            if os.path.exists(vid):
                if type == 'move':
                    move = shutil.move(vid, move_to)
                elif type == 'copy':
                    move = shutil.copy(vid, move_to)
                else:
                    raise Exception("Arg 'type' must be set to 'move or 'copy'")      
            else:
                print("Video file missing: ", vid)
        




def preprocess(file):

    print('Getting metadata....')
    metadata = get_metadata(file)
    print('Creating gloss variations....')
    gloss_variations = create_gloss_variations(metadata)
    print('Converting sample types....')
    sample_type = convert_sample_type(gloss_variations)
    print('Moving videos to folders....')
    to_folders = videos_to_folders(sample_type)


if __name__ == '__main__':
    file = 'labels/WLASL_v0.3.json'
    preprocess(file)
        