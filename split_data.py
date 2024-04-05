import os
import shutil
from random import shuffle

# Đường dẫn đến folder gốc
source_folder = './img_align_celeba/img_align_celeba'
destiny_path = './data'
# Đường dẫn đến folder train và test
train_folder = os.path.join(destiny_path, 'train')
test_folder = os.path.join(destiny_path, 'test')

# Tạo folder train và test nếu chưa tồn tại
os.makedirs(train_folder, exist_ok=True)
os.makedirs(test_folder, exist_ok=True)

# Lấy danh sách tất cả các file trong folder gốc
files = [f for f in os.listdir(source_folder) if os.path.isfile(os.path.join(source_folder, f))]

# Xáo trộn danh sách file
shuffle(files)

# Tính số lượng file cho train và test
num_train = int(len(files) * 0.8)

# # Di chuyển file vào folder train
for f in files[:num_train]:
    shutil.move(os.path.join(source_folder, f), os.path.join(train_folder, f))

# # Di chuyển file còn lại vào folder test
for f in files[num_train:]:
    shutil.move(os.path.join(source_folder, f), os.path.join(test_folder, f))
