from torch.utils.data import DataLoader
from dataset import TouchscreenSensorDataset
import torch
import torch.nn.functional as F
from utils import load_training_config_yaml
from losses.triplet_loss import OnlineTripletLoss
from losses.crs_entropy_loss import CRSEntropyLoss
import logging
import time
from datetime import datetime, date
import os
import torch.utils.data as data_utils
import torch.nn as nn

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
torch.cuda.set_device(7)

train_cfg = load_training_config_yaml()

def get_correct_gesture_cls_num(pred, label):
    
    predicted_classes = torch.argmax(pred, dim=1)
    correct_predictions = torch.eq(predicted_classes, label)
    correct_predictions_count = correct_predictions.sum().item()
    # Total predictions count is the batch size
    total_predictions_count = pred.size(0)
    return correct_predictions_count, total_predictions_count

class TrainEngine(object):
    def __init__(self, model, model_type, sensor_model_name, ts_model_name, addtional_info=""):
        self.model_type = model_type # Choose from 'gesture_classification' and 'id_recognition' and 'both'
        self.device = torch.device('cuda:0' if torch.cuda.is_available() and train_cfg['Training']['use_cuda'] else 'cpu')
        self.sensor_model_name = sensor_model_name
        self.ts_model_name = ts_model_name
        
        
        # Split dataset into train and eval
        dataset = TouchscreenSensorDataset(csv_file_path=train_cfg['Training']['data_path'], sensor_data_folder_path=train_cfg['Training']['sensor_data_path'])
        train_size = int(train_cfg['Training']['train_set_ratio'] * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = data_utils.random_split(dataset, [train_size, test_size])
        print(f'[INFO]Train dataset size: {len(train_dataset)}, Test dataset size: {len(test_dataset)}')
        
        self.train_loader = DataLoader(train_dataset, shuffle=True, batch_size=train_cfg['Training']['batch_size'])
        self.eval_loader = DataLoader(test_dataset, shuffle=True, batch_size=train_cfg['Training']['batch_size'])
        
        self.model = model
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            # self.model = nn.DataParallel(self.model)

        self.model.to(self.device)
        
        if train_cfg['Training']['optimizer'] == 'Adam':
            self.optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg['Training']['lr'], 
                                              weight_decay=train_cfg['Training']['weight_decay'])
        elif train_cfg['Training']['optimizer'] == 'SGD':
            self.optimizer = torch.optim.SGD(model.parameters(), lr=train_cfg['Training']['lr'], 
                                             weight_decay=train_cfg['Training']['weight_decay'])
        else:
            wrong_optimizer = train_cfg['Training']['optimizer']
            raise Exception(f'No optimizer named {wrong_optimizer}')
        #self.optimizer = self.optimizer.to(self.device)
        
        if model_type == 'id_only' or model_type == 'both' or model_type == 'no_gesture':
            self.loss_fn = OnlineTripletLoss(margin=train_cfg['TripletLoss']['margin'], 
                                             batch_hard=train_cfg['TripletLoss']['batch_hard'], 
                                             squared=False)
        elif model_type == 'gesture_only':
            self.loss_fn = nn.CrossEntropyLoss()
        else:
            raise Exception(f'No model_type named {model_type}')
        
        if train_cfg['Training']['lr_scheduler'] == 'StepLR':
            # self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 
            #                                                     step_size=train_cfg['Training']['step_size'],
            #                                                     gamma=train_cfg['Training']['gamma'])
            milestones = [50, 100, 1000, 2000]
            self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=milestones, gamma=train_cfg['Training']['gamma'])
        elif train_cfg['Training']['lr_scheduler'] == 'LambdaLR':
            def lr_lambda(epoch):
                if epoch < train_cfg['Training']['warmup_epochs']:
                    return epoch / train_cfg['Training']['warmup_epochs']
                else:
                    return max(0.0, 1 - (epoch - train_cfg['Training']['warmup_epochs']) / (train_cfg['Training']['epoch_num'] - train_cfg['Training']['warmup_epochs']))
            self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_lambda)
        else:
            raise Exception(f'No lr scheduler named {train_cfg["Training"]["lr_scheduler"]}')
        
        self.epoch_num = train_cfg['Training']['epoch_num']
        self.log_interval = train_cfg['Training']['log_interval']
        self.eval_interval = train_cfg['Training']['eval_interval']
        
        # Checkpoint which has the accuracy greater than 0.5 will be saved
        self.best_acc = 0.5
        self.best_ckpt_name = 'None'
        
        
        # Configure logging system
        log_path = train_cfg['Training']['log_path']
        now = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        # Get the current date in "yyyy-mm-dd" format
        log_folder_name = date.today().strftime('%Y-%m-%d')
        # Create the log folder if it does not exist
        log_folder_path = os.path.join(log_path, log_folder_name)
        if not os.path.exists(log_folder_path):
            os.makedirs(log_folder_path)
        
        self.logger = logging.getLogger('Training')
        self.logger.setLevel(level=logging.DEBUG)
        
        handler = logging.FileHandler(filename=f'{log_folder_path}/{model_type}_{now}.log', encoding='utf-8', mode='w')
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        
        self.logger.info(f'[INFO]Train dataset size: {len(train_dataset)}, Test dataset size: {len(test_dataset)}')
        self.logger.info(f'[INFO]Model type: {model_type}, Sensor model name: {sensor_model_name}, TS model name: {ts_model_name}')
        self.logger.info(addtional_info)
        
    def eval(self):
        self.model.eval()
        eval_stats = dict()
        eval_loss = 0.0
        correct_pairs = 0
        total_pairs = 0
        for batch_idx, batch in enumerate(self.eval_loader):
            # sensor_data, ts_data, total_time, gesture_type, usrn_pswd_len, label = batch
            # sensor_data = sensor_data.to(self.device)
            # for data in ts_data:
            #     data.to(self.device)
            # total_time, usrn_pswd_len = total_time.to(self.device), usrn_pswd_len.to(self.device)
            # gesture_type, label = gesture_type.to(self.device), label.to(self.device)
            
            sensor_data, ts_data, total_time, gesture_type, usrn_pswd_len, label = batch
            sensor_data = sensor_data.to(self.device)
            
            ts_data = [data.to(self.device) for data in ts_data]
            # for data in ts_data:
            #     data = data.to(self.device)
            total_time, usrn_pswd_len = total_time.to(self.device), usrn_pswd_len.to(self.device)
            gesture_type, label = gesture_type.to(self.device), label.to(self.device)
            
            # Convert gesture_type to one-hot vector
            # gesture_type = F.one_hot(gesture_type, num_classes=8).float()
            if self.model_type == 'gesture_only':
                # embeddings.shape = (batch_size, embedding_size)
                embeddings = self.model(sensor_data, ts_data, total_time, usrn_pswd_len)
                loss = self.loss_fn(embeddings, gesture_type)
                batch_correct, batch_samples_num = get_correct_gesture_cls_num(embeddings, gesture_type)
                no_positive_anchor_num = 0
            elif self.model_type == 'id_only':
                # embeddings.shape = (batch_size, embedding_size)
                embeddings = self.model(ts_data, gesture_type, total_time, usrn_pswd_len)
                loss, batch_correct, batch_samples_num, no_positive_anchor_num = self.loss_fn(embeddings, label)
            elif self.model_type == 'both':
                embeddings = self.model(sensor_data, ts_data, total_time, usrn_pswd_len)
                loss, batch_correct, batch_samples_num, no_positive_anchor_num = self.loss_fn(embeddings, label)
            elif self.model_type == 'no_gesture':
                embeddings = self.model(sensor_data, ts_data, total_time, usrn_pswd_len)
                loss, batch_correct, batch_samples_num, no_positive_anchor_num = self.loss_fn(embeddings, label)
            
            eval_loss += loss.item()
            correct_pairs += batch_correct
            total_pairs += (batch_samples_num - no_positive_anchor_num)
        eval_stats['loss'] = eval_loss / (batch_idx+1)
        eval_stats['correct_pairs'] = correct_pairs
        eval_stats['total_pairs'] = total_pairs
        return eval_stats
    
    def save_checkpoint(self, save_path, epoch, save_all=False, use_time=True):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if save_all:
            save_dict = {'epoch': epoch, 'model_type': self.model_type,
                         'model': self.model.state_dict(),
                         'optimizer': self.optimizer.state_dict(),
                         'lr_scheduler': self.lr_scheduler.state_dict()}
        else:
            save_dict = self.model.state_dict()
        if use_time:
            current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            ckpt_save_name = f'{save_path}/{self.model.__class__.__name__}_{current_time}.pth'
        else:
            ckpt_save_name = f'{save_path}/{self.model.__class__.__name__}.pth'
        torch.save(save_dict, ckpt_save_name)
        # Lastly saved checkpoint is the best checkpoint
        self.best_ckpt_name = ckpt_save_name.split('/')[-1]
    
    def train_one_epoch(self, current_epoch, total_epoch):
        train_epoch_stats = dict()
        epoch_total_samples = 0
        epoch_correct_samples = 0
        epoch_loss = 0.0
        for batch_idx, batch in enumerate(self.train_loader):
            self.model.train()
            self.optimizer.zero_grad()
            
            sensor_data, ts_data, total_time, gesture_type, usrn_pswd_len, label = batch
            sensor_data = sensor_data.to(self.device)
            
            ts_data = [data.to(self.device) for data in ts_data]
            # for data in ts_data:
            #     data = data.to(self.device)
            total_time, usrn_pswd_len = total_time.to(self.device), usrn_pswd_len.to(self.device)
            gesture_type, label = gesture_type.to(self.device), label.to(self.device)

            # Convert gesture_type to one-hot vector
            # gesture_type = F.one_hot(gesture_type, num_classes=8).float()
            if self.model_type == 'gesture_only':
                # embeddings.shape = (batch_size, embedding_size)
                embeddings = self.model(sensor_data, ts_data, total_time, usrn_pswd_len)
                loss = self.loss_fn(embeddings, gesture_type)
                batch_correct, batch_samples_num = get_correct_gesture_cls_num(embeddings, gesture_type)
                no_positive_anchor_num = 0
            elif self.model_type == 'id_only':
                # embeddings.shape = (batch_size, embedding_size)
                embeddings = self.model(ts_data, gesture_type, total_time, usrn_pswd_len)
                loss, batch_correct, batch_samples_num, no_positive_anchor_num = self.loss_fn(embeddings, label)
            elif self.model_type == 'both':
                embeddings = self.model(sensor_data, ts_data, total_time, usrn_pswd_len)
                loss, batch_correct, batch_samples_num, no_positive_anchor_num = self.loss_fn(embeddings, label)
            elif self.model_type == 'no_gesture':
                embeddings = self.model(sensor_data, ts_data, total_time, usrn_pswd_len)
                loss, batch_correct, batch_samples_num, no_positive_anchor_num = self.loss_fn(embeddings, label)
            
             # Check if loss is NaN
            if torch.isnan(loss):
                print('Loss is NaN!')
                print('embeddings:', embeddings)
                print('gesture_type:', gesture_type)
                print(f'Learning Rate: {self.lr_scheduler.get_last_lr()[0]:.4f}')
                break  # Exit the loop as continuing training might be futile
            
            
            epoch_total_samples += (batch_samples_num - no_positive_anchor_num)
            epoch_correct_samples += batch_correct
            
            epoch_loss += loss.item()
            
            loss.backward()
            self.optimizer.step()
            
            
            # Log the training information
            if batch_idx % self.log_interval == 0:
                print(f'[TRAIN]EPOCH: {current_epoch}/{total_epoch}, batch: {batch_idx}, loss: {loss.item():.4f}, lr: {self.lr_scheduler.get_last_lr()[0]:.4f}')
                self.logger.info(f'[TRAIN]EPOCH: {current_epoch}/{total_epoch}, batch: {batch_idx}, loss: {loss.item():.4f}, lr: {self.lr_scheduler.get_last_lr()[0]:.4f}')
            
            # Evaluate the model, save the model and log the evaluation information
            if batch_idx % self.eval_interval == 0:
                eval_stats = self.eval()
                acc = eval_stats['correct_pairs'] / eval_stats['total_pairs']
                print(f'[EVAL]EPOCH: {current_epoch}/{total_epoch}, batch: {batch_idx}, loss: {eval_stats["loss"]:.4f}, accuracy: {eval_stats["correct_pairs"]}/{eval_stats["total_pairs"]}, {acc:.4f}')
                self.logger.info(f'[EVAL]EPOCH: {current_epoch}/{total_epoch}, batch: {batch_idx}, loss: {eval_stats["loss"]:.4f}, accuracy: {eval_stats["correct_pairs"]}/{eval_stats["total_pairs"]}, {acc:.4f}')
                if acc > self.best_acc:
                    self.best_acc = acc
                    self.save_checkpoint(save_path=train_cfg['Training']['checkpoint_path'], epoch=current_epoch)
                    print(f'[SAVE]EPOCH: {current_epoch}/{total_epoch}, batch: {batch_idx}, best_acc: {self.best_acc}, best_ckpt: {self.best_ckpt_name}')
                    self.logger.info(f'[SAVE]EPOCH: {current_epoch}/{total_epoch}, batch: {batch_idx}, best_acc: {self.best_acc}, best_ckpt: {self.best_ckpt_name}')
        self.lr_scheduler.step()
        epoch_loss /= (batch_idx+1)
        return epoch_loss, epoch_correct_samples, epoch_total_samples
    
    def train(self):
        self.model.train()
            
        for epoch in range(self.epoch_num):
            start_time = time.time()
            epoch_loss, epoch_correct_samples, epoch_total_samples = self.train_one_epoch(current_epoch=epoch, total_epoch=self.epoch_num)
            epoch_time = time.time() - start_time
            epoch_time = time.strftime('%H:%M:%S', time.gmtime(epoch_time))
            print(f'[TRAIN]EPOCH: {epoch}/{self.epoch_num}, Epoch Time: {epoch_time}, Epoch Loss: {epoch_loss:.4f}, Epoch Accuracy: {epoch_correct_samples}/{epoch_total_samples}')
            self.logger.info(f'[TRAIN]EPOCH: {epoch}/{self.epoch_num}, Epoch Time: {epoch_time}, Epoch Loss: {epoch_loss:.4f}, Epoch Accuracy: {epoch_correct_samples}/{epoch_total_samples}')
            
            # Eval model at the end of each epoch
            eval_stats = self.eval()
            acc = eval_stats['correct_pairs'] / eval_stats['total_pairs']
            print(f'[EVAL]EPOCH: {epoch}/{self.epoch_num}, loss: {eval_stats["loss"]:.4f}, accuracy: {eval_stats["correct_pairs"]}/{eval_stats["total_pairs"]}, {acc:.4f}')
            self.logger.info(f'[EVAL]EPOCH: {epoch}/{self.epoch_num}, loss: {eval_stats["loss"]:.4f}, accuracy: {eval_stats["correct_pairs"]}/{eval_stats["total_pairs"]}, {acc:.4f}')
            if acc > self.best_acc:
                self.best_acc = acc
                self.save_checkpoint(save_path=train_cfg['Training']['checkpoint_path'], epoch=epoch)
                print(f'[SAVE]EPOCH: {epoch}/{self.epoch_num}, best_acc: {self.best_acc}, best_ckpt: {self.best_ckpt_name}')
                self.logger.info(f'[SAVE]EPOCH: {epoch}/{self.epoch_num}, best_acc: {self.best_acc}, best_ckpt: {self.best_ckpt_name}')
            
        print(f'[DONE]Best_acc: {self.best_acc}, Best_ckpt: {self.best_ckpt_name}')
        self.logger.info(f'[DONE]Best_acc: {self.best_acc}, Best_ckpt: {self.best_ckpt_name}')
        return

    
    
        
        