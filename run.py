import os
import shutil

# data_split_txt = "/media/home/zyzeng/workspace/WorDepth_Swin/data_splits/eigen_test_files_with_gt.txt"
data_split_txt = "/media/home/zyzeng/workspace/WorDepth_Swin/data_splits/nyudepthv2_train_files_with_gt.txt"

# source_folder = "/media/staging1/zyzeng/kitti_raw_data/"
source_folder = "/media/staging1/zyzeng/nyu_depth_v2/sync/"
# source_folder = "/media/staging1/zyzeng/nyu_depth_v2/official_splits/test/"

# save_dir = "/media/home/zyzeng/workspace/WorDepth_Swin/text_feat/kitti/test/"
save_dir = "/media/home/zyzeng/workspace/WorDepth_Swin/text_feat/nyu/train/"
count = 0
with open(data_split_txt, 'r') as file:
    lines = file.readlines()
# Iterate over each line in the datasplit.txt
for line in lines:
    count += 1
    print("Count: ", count)
    image_path = line.split(" ")[0]
    text_path = image_path[:-4]+".pt"
    # print(image_path)
    folder = image_path[0:image_path.rfind('/')]
    save_name = image_path.split("/")[-1].split(".")[0]+".pt"
    # print(save_name)
    if text_path[0] == "/":
        text_path = text_path[1:]
    full_text_path = os.path.join(source_folder, text_path)
    # print(full_image_path)

    if not os.path.exists(save_dir+folder):
        os.makedirs(save_dir+folder)

    full_save_dir = save_dir+folder
    full_save_name = save_dir+folder+"/"+save_name

    # print(full_text_path)
    # print(full_save_name)
    shutil.copy(full_text_path, full_save_name)