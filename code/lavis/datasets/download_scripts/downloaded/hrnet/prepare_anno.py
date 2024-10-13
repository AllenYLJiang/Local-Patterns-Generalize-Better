import os 
import shutil
import json
import joblib

ref_anno_json = json.load(open('/root/data/LAVIS-main/lavis/datasets/download_scripts/downloaded/vg/annotations/vg_caption.json'))
# this is the captions
caption_dir = '/root/data/Anomaly_detection_921/STG-NF/data/ShanghaiTech/images/image_hrnet_blip2caption/captions_new'
# this is the concatenation of 17-channel heatmaps and 3-channel human images
heatmap_dir = '/root/data/Anomaly_detection_921/STG-NF/data/ShanghaiTech/images/image_hrnet_blip2caption/heatmaps_new'
# this is the visualization of human images
human_region_vis_dir = '/root/data/Anomaly_detection_921/STG-NF/data/ShanghaiTech/images/image_hrnet_blip2caption/images_new'

anno_list = []
image_id = 1
for img_tensor_name in os.listdir(heatmap_dir):
    curr_img_contents = {
        'image': os.path.join(heatmap_dir, img_tensor_name),
        'caption': joblib.load(os.path.join(caption_dir, img_tensor_name)),
        'image_id': 'hrnet_' + str(image_id),
        'dataset': 'hrnet'
    }
    anno_list.append(curr_img_contents)
    image_id += 1

out_file = open("/root/data/LAVIS-main/lavis/datasets/download_scripts/downloaded/hrnet/annotations/hrnet_caption.json", "w")
json.dump(anno_list, out_file)
out_file.close()