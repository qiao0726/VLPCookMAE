from torch.utils.data import Dataset
import torch
from utils import load_csv_to_list, encode_non_numeric
import ast
import os
from pprint import pprint


# When data length is < 30, we will pad the data with 0s
# if > 30, we will truncate the data
FIXED_INPUT_LENGTH = 30
def set_seq_len(input_seq:list, cut_start=True, pad_start=True):
    if len(input_seq) < FIXED_INPUT_LENGTH:
        if pad_start:
            input_seq = [0] * (FIXED_INPUT_LENGTH - len(input_seq)) + input_seq
        else:
            input_seq = input_seq + [0] * (FIXED_INPUT_LENGTH - len(input_seq))
    elif len(input_seq) > FIXED_INPUT_LENGTH:
        if cut_start:
            input_seq = input_seq[-FIXED_INPUT_LENGTH:]
        else:
            input_seq = input_seq[:FIXED_INPUT_LENGTH]
    return input_seq


FIXED_SENSOR_DATA_LENGTH = 500
def set_sensor_data_len(sensor_data:list, cut_start=True, pad_start=True):
    return_list = list()
    # if sensor_data's length is less than FIXED_SENSOR_DATA_LENGTH, pad it with 0s
    if len(sensor_data) < FIXED_SENSOR_DATA_LENGTH:
        for i in range(FIXED_SENSOR_DATA_LENGTH - len(sensor_data)):
            return_list.append([0,0,0,0,0,0])
    
    for this_moment_sensor_data in sensor_data:
        accX,accY,accZ = this_moment_sensor_data['accX'],this_moment_sensor_data['accY'],this_moment_sensor_data['accZ']
        gyroX,gyroY,gyroZ = this_moment_sensor_data['gyroX'],this_moment_sensor_data['gyroY'],this_moment_sensor_data['gyroZ']
        return_list.append([accX,accY,accZ,gyroX,gyroY,gyroZ])
        
    # if sensor_data's length is greater than FIXED_SENSOR_DATA_LENGTH, cut it
    if len(sensor_data) > FIXED_SENSOR_DATA_LENGTH:
        return_list = return_list[-FIXED_SENSOR_DATA_LENGTH:]
        
    return return_list
    

class TouchscreenDataset(Dataset):
    def __init__(self, csv_file_path,
                 applied_data_list=('hold_time', 'inter_time', 'distance', 'speed')):
        self.data, self.label_dict = load_csv_to_list(csv_file_path)
        
        self.apply_data_list = applied_data_list
        
    def load_one_item(self, index):
        item = self.data[index]
        if 'hold_time' in self.apply_data_list:
            hold_time = list(ast.literal_eval(item['hold-time']))
            hold_time = set_seq_len(hold_time)
            hold_time = torch.tensor(hold_time, dtype=torch.float32)
            #hold_time = hold_time[:FIXED_INPUT_LENGTH] if len(hold_time) > FIXED_INPUT_LENGTH else hold_time + [0] * (FIXED_INPUT_LENGTH - len(hold_time))
            #item['hold_time'] = hold_time
        if 'inter_time' in self.apply_data_list:
            inter_time = list(ast.literal_eval(item['inter-time']))
            inter_time = set_seq_len(inter_time)
            inter_time = torch.tensor(inter_time, dtype=torch.float32)
        # if 'pressure' in self.apply_data_list:
        #     pressure = list(ast.literal_eval(item['pressure']))
        #     pressure = torch.tensor(pressure, dtype=torch.float32)
        if 'distance' in self.apply_data_list:
            distance = list(ast.literal_eval(item['distance']))
            distance = set_seq_len(distance)
            distance = torch.tensor(distance, dtype=torch.float32)
        # Compute speed base on distance and inter_time
        if 'speed' in self.apply_data_list:
            speed = list()
            for a, b in zip(distance, inter_time):
                if b == 0:
                    speed.append(0)
                else:
                    speed.append(a/b)
            speed = torch.tensor(speed, dtype=torch.float32)
            
        # Convert to torch.tensor
        total_time = torch.tensor(item['total_time'], dtype=torch.float32)
        
        # Get gesture type
        # ation：1 for standing, 2 for walking, 3 for lying
        # pose:
        # 1 for one-handed holding, one-handed operation, 2 for one-handed holding, the other hand operation,
        # 3 for two-handed holding, two-handed operation, 4 for device on table, one-handed operation, 
        # 5 for device on table, two-handed operation
        action = int(item['action'])
        pose = int(item['pose'])
        gesture_type_map = [
            [0,2,4,6,7],
            [1,3,5]
        ]
        
        gesture_type = gesture_type_map[action-1][pose-1]
        #gesture_type = torch.tensor(gesture_type, dtype=torch.float32)
        
        # label = encode_non_numeric(str(item['label']))
        label = self.label_dict[item['label']]
        
        usrn_psrd_len = torch.tensor(item['usrn_len'] + item['pswd_len'], dtype=torch.float32)
        
        return label, hold_time, inter_time, distance, speed, total_time, gesture_type, usrn_psrd_len
    
    def get_len(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.load_one_item(index)

    def __len__(self):
        return len(self.data)


class SensorDataset(Dataset):
    def __init__(self, csv_file_path, sensor_data_folder_path):
        self.login_data, _ = load_csv_to_list(csv_file_path)
        self.sensor_data_folder_path = sensor_data_folder_path
    
    def load_one_item(self, index):
        this_item = self.login_data[index]
        uuid = this_item['uuid']
        
        # Find the sensor data file of this uuid
        sensor_data_csv_file_path = os.path.join(self.sensor_data_folder_path, f'{uuid}.csv')
        sensor_data, _ = load_csv_to_list(sensor_data_csv_file_path)
        
        # sensor_data in each sample might have different length
        # So we need to pad or truncate them to the same length
        return_list = set_sensor_data_len(sensor_data)
        # Convert to torch.tensor
        sensor_data_tensor = torch.tensor(return_list, dtype=torch.float32)
        
        action = int(this_item['action'])
        pose = int(this_item['pose'])
        gesture_type_map = [
            [0,2,4,6,7],
            [1,3,5]
        ]
        gesture_type = gesture_type_map[action-1][pose-1]
        #gesture_type = torch.tensor(gesture_type, dtype=torch.float32)
        
        total_time = torch.tensor(this_item['total_time'], dtype=torch.float32)
        
        label = encode_non_numeric(str(this_item['label']))
        
        usrn_psrd_len = torch.tensor(this_item['usrn_len'] + this_item['pswd_len'], dtype=torch.float32)
        
        return sensor_data_tensor, int(label), total_time, gesture_type, usrn_psrd_len
    
    def get_len(self):
        return len(self.login_data)
    
    def __len__(self):
        return len(self.login_data)
    
    def __getitem__(self, index):
        self.load_one_item(index)
        

class TouchscreenSensorDataset(Dataset):
    def __init__(self, csv_file_path, sensor_data_folder_path):
        # self.touchscreen_ds = TouchscreenDataset(csv_file_path)
        # self.sensor_ds = SensorDataset(csv_file_path, sensor_data_folder_path)
        # touchscreen data
        self.data, self.label_dict = load_csv_to_list(csv_file_path)
        self.apply_data_list = ('hold_time', 'inter_time', 'distance', 'speed')
        # sensor data
        self.sensor_data_folder_path = sensor_data_folder_path
        
    
    def __getitem__(self, index):
        uuid, label, hold_time, inter_time, distance, speed, total_time, gesture_type, usrn_pswd_len = self.load_touchscreen_data(index)
        ts_data = (hold_time, inter_time, distance, speed)
        # print(total_time.shape)
        # print(usrn_pswd_len.shape)
        # print(label.shape)
        total_time, usrn_pswd_len, label = total_time.unsqueeze(0), usrn_pswd_len.unsqueeze(0), label.unsqueeze(0)
        
        sensor_data_tensor = self.load_sensor_data(uuid)
        return (sensor_data_tensor, ts_data, total_time, gesture_type, usrn_pswd_len, label)
    
    def __len__(self):
        return len(self.data)
    
    
    def load_sensor_data(self, uuid):
        
        # Find the sensor data file of this uuid
        sensor_data_csv_file_path = os.path.join(self.sensor_data_folder_path, f'{uuid}.csv')
        sensor_data, _ = load_csv_to_list(sensor_data_csv_file_path)
        
        # sensor_data in each sample might have different length
        # So we need to pad or truncate them to the same length
        return_list = set_sensor_data_len(sensor_data)
        # Convert to torch.tensor
        sensor_data_tensor = torch.tensor(return_list, dtype=torch.float32)
        
        return sensor_data_tensor
    
    def load_touchscreen_data(self, index):
        item = self.data[index]
        if 'hold_time' in self.apply_data_list:
            hold_time = list(ast.literal_eval(item['hold-time']))
            hold_time = set_seq_len(hold_time)
            hold_time = torch.tensor(hold_time, dtype=torch.float32)
            #hold_time = hold_time[:FIXED_INPUT_LENGTH] if len(hold_time) > FIXED_INPUT_LENGTH else hold_time + [0] * (FIXED_INPUT_LENGTH - len(hold_time))
            #item['hold_time'] = hold_time
        if 'inter_time' in self.apply_data_list:
            inter_time = list(ast.literal_eval(item['inter-time']))
            inter_time = set_seq_len(inter_time)
            inter_time = torch.tensor(inter_time, dtype=torch.float32)
        # if 'pressure' in self.apply_data_list:
        #     pressure = list(ast.literal_eval(item['pressure']))
        #     pressure = torch.tensor(pressure, dtype=torch.float32)
        if 'distance' in self.apply_data_list:
            distance = list(ast.literal_eval(item['distance']))
            distance = set_seq_len(distance)
            distance = torch.tensor(distance, dtype=torch.float32)
        # Compute speed base on distance and inter_time
        if 'speed' in self.apply_data_list:
            speed = list()
            for a, b in zip(distance, inter_time):
                if b == 0:
                    speed.append(0)
                else:
                    speed.append(a/b)
            speed = torch.tensor(speed, dtype=torch.float32)
            
        # Convert to torch.tensor
        total_time = torch.tensor(item['total_time'], dtype=torch.float32)
        
        # Get gesture type
        # ation：1 for standing, 2 for walking, 3 for lying
        # pose:
        # 1 for one-handed holding, one-handed operation, 2 for one-handed holding, the other hand operation,
        # 3 for two-handed holding, two-handed operation, 4 for device on table, one-handed operation, 
        # 5 for device on table, two-handed operation
        action = int(item['action'])
        pose = int(item['pose'])
        gesture_type_map = [
            [0,2,4,6,7],
            [1,3,5]
        ]
        
        gesture_type = gesture_type_map[action-1][pose-1]
        gesture_type = torch.tensor(gesture_type, dtype=torch.int64)
        
        # label = encode_non_numeric(str(item['label']))
        label = self.label_dict[item['label']]
        label = torch.tensor(label, dtype=torch.float32)
        
        usrn_psrd_len = torch.tensor(item['usrn_len'] + item['pswd_len'], dtype=torch.float32)
        
        return item['uuid'], label, hold_time, inter_time, distance, speed, total_time, gesture_type, usrn_psrd_len


    
class TripletTouchscreenDataset(Dataset):
    def __init__(self, touchscreen_ds: TouchscreenDataset, train=True):
        self.dataset = touchscreen_ds
        self.train = train
        
        # Prepare triplets
        triplets = []
        for index in range(self.dataset.get_len()):
            anchor = self.dataset.load_one_item(index)
            anchor_id = anchor[0]
            anchor_label = anchor[1]
            # find a positive example for this anchor
            for positive_index in range(self.dataset.get_len()):
                positive = self.dataset.load_one_item(positive_index)
                # if this example has different label with anchor, skip it
                if positive[1] != anchor_label or positive[0] == anchor_id:
                    continue
                
                # find a negative example for this anchor
                for negative_index in range(self.dataset.get_len()):
                    negative = self.dataset.load_one_item(negative_index)
                    # if this example has same label with anchor, skip it
                    if negative[1] == anchor_label:
                        continue
                    # if this example has same id with anchor, skip it
                    if negative[0] == anchor_id:
                        continue
                    # negative example found
                    triplets.append((anchor, positive, negative))
        self.triplets = triplets

    def __getitem__(self, index):
        triplet = self.triplets[index]
        return triplet

    def __len__(self):
        return len(self.triplets)
    
    
