import os
import json
import joblib
import orjson

src_dir = '/path/to/train_data_SMM'
dst_dir = '/path/to/train_data_SMM'

for video_name in os.listdir(src_dir):
    curr_video_dst_format = {}

    curr_video_info = joblib.load(os.path.join(src_dir, video_name))
    num_ids = len(curr_video_info)
    for ID_idx in range(num_ids):
        curr_video_dst_format[str(ID_idx+1)] = {}
        curr_ID_info = curr_video_info[ID_idx]
        for time_idx in range(len(curr_ID_info)):
            curr_video_dst_format[str(ID_idx+1)]['%04d'%time_idx] = {
                'keypoints': curr_ID_info[time_idx].tolist(),
                'scores': 1.0
            }

    # out_file = open(os.path.join(dst_dir, video_name.split('.')[0] + '.json'), "w")

    with open(os.path.join(dst_dir, '01_' + video_name.split('.')[0].split('_')[-1] + '_alphapose_tracked_person.json'), "wb") as f:
        f.write(orjson.dumps(curr_video_dst_format))

    # json.dump(curr_video_dst_format, out_file)
    # out_file.close()

