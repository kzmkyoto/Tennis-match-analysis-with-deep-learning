import os
import pandas as pd
import pathlib as Path


def create_gt_labels(root_dir=r'c:/kyoto/TennisML/TrackNet/Dataset', path_csv_output=r'c:/kyoto/TennisML/TrackNet/Dataset', train_propotion=0.7):
    # Merge label.csv files in each end of clips to one csv file

    df = pd.DataFrame()

    for game_id in range(1, 11):   
        path_game = os.path.join(root_dir, f"game{game_id}")
        clips = os.listdir(path_game)
        for clip in clips:
            labels = pd.read_csv(os.path.join(path_game, clip, 'Label.csv'))
            labels['path_now'] = labels['file name'].apply(lambda k: os.path.join(f"game{game_id}", clip, k))

            # labels_gt = labels.iloc[2:].copy()      # remove the first two rows
            labels_gt = labels.iloc[:].copy()
            labels_gt.loc[:, 'path_prev'] = labels['path_now'].shift(1)
            labels_gt.loc[:, 'path_prevprev'] = labels['path_now'].shift(2)
            labels_gt.loc[:, 'path_gt'] = labels['path_now']
            labels_gt = labels_gt[2:]
            df = pd.concat([df, labels_gt], ignore_index=True)

    df = df.loc[:, ['path_now', 'path_prev', 'path_prevprev', 'path_gt', 'x-coordinate', 'y-coordinate', 'status', 'visibility']]
    # Replace backslashes with forward slashes in all path columns
    for col in ['path_now', 'path_prev', 'path_prevprev', 'path_gt']:
        df[col] = df[col].str.replace('\\', '/', regex=False)

    num_train = int(len(df.index)*train_propotion)
    df_train = df.loc[:num_train]
    df_val = df.loc[num_train:]
    df_train.to_csv(os.path.join(path_csv_output, 'labels_train.csv'), index=False)
    df_val.to_csv(os.path.join(path_csv_output, 'labels_val.csv'), index=False)
    # print(df_train)


# def create_gt_labels_2(path_input='c:/kyoto/TennisML/TrackNet/Dataset', path_output='c:/kyoto/TennisML/TrackNet/Dataset', train_rate=0.7):
#     df = pd.DataFrame()
#     for game_id in range(1,11):
#         game = 'game{}'.format(game_id)
#         clips = os.listdir(os.path.join(path_input, game))
#         for clip in clips:
#             labels = pd.read_csv(os.path.join(path_input, game, clip, 'Label.csv'))
#             labels['gt_path'] = 'gts/' + game + '/' + clip + '/' + labels['file name']
#             labels['path1'] = 'images/' + game + '/' + clip + '/' + labels['file name']
#             labels_target = labels[2:]
#             labels_target.loc[:, 'path2'] = list(labels['path1'][1:-1])
#             labels_target.loc[:, 'path3'] = list(labels['path1'][:-2])
#             df = pd.concat([df, labels_target], ignore_index=True)
#     df = df.reset_index(drop=True)
#     df = df[['path1', 'path2', 'path3', 'gt_path', 'x-coordinate', 'y-coordinate', 'status', 'visibility']]
#     # df = df.sample(frac=1)
#     num_train = int(df.shape[0]*train_rate)
#     df_train = df[:num_train]
#     df_test = df[num_train:]
#     df_train.to_csv(os.path.join(path_output, 'labels_train.csv'), index=False)
#     df_test.to_csv(os.path.join(path_output, 'labels_val.csv'), index=False)

if __name__ == '__main__':
    create_gt_labels()
