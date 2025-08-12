import os
import pandas as pd
import numpy as np
import cv2


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


def gaussian_kernel(size, sigma):
    '''
    sigma(variance) is equivalent to the average radius of a tennis ball (about 5 pixels)
    '''
    x, y = np.mgrid[-size:size+1, -size:size+1]
    gk = np.exp( -(x**2 + y**2) / (float(2*sigma)) )
    return gk

def generate_gaussian_kernel_array(size, sigma):
    gaussian_kernel_array = gaussian_kernel(size, sigma)
    gaussian_kernel_array = gaussian_kernel_array * (255 / gaussian_kernel_array[ int(len(gaussian_kernel_array)/2) ][ int(len(gaussian_kernel_array)/2) ])
    gaussian_kernel_array = gaussian_kernel_array.astype(int)
    return gaussian_kernel_array


def create_gt_images(size, sigma, width, height, root_dir='c:/kyoto/TennisML/TrackNet/Dataset', path_output='c:/kyoto/TennisML/TrackNet/Dataset/gts'):
    for game_id in range(1, 11):
        path_game = os.path.join(root_dir, f"game{game_id}")
        clips = os.listdir(path_game)

        path_output_game = os.path.join(path_output, f"game{game_id}")
        if not os.path.exists(path_output_game):
            os.makedirs(path_output_game)

        for clip in clips:
            path_output_clip = os.path.join(path_output_game, clip)
            if not os.path.exists(path_output_clip):
                os.makedirs(path_output_clip)

            labels = pd.read_csv(os.path.join(path_game, clip, 'Label.csv'))
            for idx in range(len(labels.index)):
                file_name, visibility, x, y, _ = labels.loc[idx, :]
                heatmap = np.zeros((width, height, 3), dtype=np.uint8)
                if visibility != 0:
                    x, y = int(x), int(y)
                    for i in range(-size, size+1):
                        for j in range(-size, size+1):
                            if x+i >= 0 and x+i < width and y+j >= 0 and y+j < height:
                                gaussian_kernel_array = generate_gaussian_kernel_array(size, sigma)
                                temp = gaussian_kernel_array[size+i][size+j]
                                if temp > 0:
                                    heatmap[x+i][y+j] = (temp, temp, temp)
                cv2.imwrite(os.path.join(path_output_clip, file_name), heatmap)


def postprocess(feature_map, scale=2, shape=(360, 640), threshold=127, min_radius=2, max_radius=7):
    feature_map = np.array(feature_map)
    feature_map = (feature_map*255).astype(np.uint8)
    feature_map = feature_map.reshape(shape)

    _, heatmap = cv2.threshold(feature_map, threshold, 255, cv2.THRESH_BINARY)
    circles = cv2.HoughCircles(heatmap, cv2.HOUGH_GRADIENT, dp=1, minDist=1, param1=50, param2=2, minRadius=min_radius, maxRadius=max_radius)

    x, y = None, None
    if circles is not None and circles.shape[1] > 0:
        best_circle = circles[0][0]
        x = best_circle[0] * scale
        y = best_circle[0] * scale
    return x, y

