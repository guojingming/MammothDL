import os
import utils.shapenet_provider as sp
import utils.matplot_viewer as mpv

# Download shapenet data
sp.download_data()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/')
TRAIN_FILES = sp.getDataFiles(os.path.join(DATA_DIR, 'train_files.txt'))
LABLE_NAMES = sp.getLabelNameDictionary(os.path.join(DATA_DIR, 'shape_names.txt'))
for i in range(len(TRAIN_FILES)):
    TRAIN_FILES[i] = os.path.join(BASE_DIR, TRAIN_FILES[i])

# print(TRAIN_FILES)

for i in range(len(TRAIN_FILES)):
    temp_data = sp.load_h5(TRAIN_FILES[i])
    temp_labels = temp_data[1]
    # print(temp_labels)
    for j in range(3, len(temp_data[0])):
        mpv.show_pointcloud_fromarray(temp_data[0][j], LABLE_NAMES[temp_labels[j][0]])
        # input()
