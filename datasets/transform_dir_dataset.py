import os, cv2
import numpy as np
from tqdm import tqdm


path = 'images'
vid_counter = 0
img_counter = 0
car_counter = 0
cam_counter = 0
camera_dict = {
    'OrderWindow2': 0,
    'PayWindow': 1,
    'PickupWindow': 2
}

os.system('rm -rf train_test_split')
os.makedirs('train_test_split')


train = list()
test_small, test_small_q = list(), list()
test_medium, test_medium_q = list(), list()
test_large, test_large_q = list(), list()
useless_data = [('path', 'cam_id', '1', '2', '3', '4')]

np.random.seed(0)


def save_data(data, path):
    with open(os.path.join('train_test_split', path), 'w') as file:
        for row in data:
            file.write(' '.join(map(str, row)) + '\n')


def save_dir(path, hard_test=False):
    global car_counter, img_counter
    is_test = np.random.rand() < 0.075
    files = os.listdir(path)
    np.random.shuffle(files)

    if is_test or hard_test:
        random_cam = np.random.randint(3)
        is_test_small = random_cam == 0

    for num, frame in enumerate(files):
        img_counter += 1
        cam_num = camera_dict[frame.split('_')[0]]
        img_path = os.path.join(str(vid_counter), str(img_counter))
        useless_data.append((img_path, cam_counter + cam_num, 1, 2, 3, 4))
        img_path = f'{img_path}.jpg'
        frame_data = (img_path, car_counter, cam_counter + cam_num)
        os.system(f'mv {os.path.join(path, frame)} images/{img_path}')
        
        if hard_test:
            if cam_num == random_cam:
                test_large_q.append(frame_data)
            else:
                test_large.append(frame_data)
        elif is_test:
            if cam_num == random_cam:
                test_medium_q.append(frame_data)
                if is_test_small:
                    test_small_q.append(frame_data)
            else:
                test_medium.append(frame_data)
                if is_test_small:
                    test_small.append(frame_data)
        else:
            train.append(frame_data)

    car_counter += 1


for store in os.listdir(path):
    os.makedirs(os.path.join('images', str(vid_counter)))
    hard_test = store in ['51589', '21148', '40285', '61631']
    cars = os.listdir(os.path.join(path, store))
    np.random.shuffle(cars)
    for car in tqdm(cars):
        save_dir(os.path.join(path, store, car), hard_test)
    cam_counter += 3
    vid_counter += 1
    
    
save_data(train, 'train_list.txt')
save_data(test_small, 'test_3000.txt')
save_data(test_medium, 'test_5000.txt')
save_data(test_large, 'test_10000.txt')
save_data(test_small_q, 'test_3000_query.txt')
save_data(test_medium_q, 'test_5000_query.txt')
save_data(test_large_q, 'test_10000_query.txt')

with open(os.path.join('train_test_split', 'vehicle_info.txt'), 'w') as file:
    for row in useless_data:
        file.write(';'.join(map(str, row)) + '\n')

