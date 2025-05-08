import os
import torch
import numpy as np
import pandas as pd
import math
import re
import pdb
import pickle
import time
from scipy import stats


from torch.utils.data import Dataset
import h5py

from utils_clam.utils import generate_split, nth

def safe_open_h5(path, retry=3, delay=0.5):
    for attempt in range(retry):
        try:
            return h5py.File(path, 'r')
        except OSError as e:
            print(f"[WARN] Failed to open {path} (attempt {attempt+1}/{retry}): {e}")
            time.sleep(delay)
    raise OSError(f"[FATAL] Failed to open {path} after {retry} attempts")

def save_splits(split_datasets, column_keys, filename, boolean_style=False):
	splits = [split_datasets[i].slide_data['slide_id'] for i in range(len(split_datasets))]
	if not boolean_style:
		df = pd.concat(splits, ignore_index=True, axis=1)
		df.columns = column_keys
	else:
		df = pd.concat(splits, ignore_index = True, axis=0)
		index = df.values.tolist()
		one_hot = np.eye(len(split_datasets)).astype(bool)
		bool_array = np.repeat(one_hot, [len(dset) for dset in split_datasets], axis=0)
		df = pd.DataFrame(bool_array, index=index, columns = ['train', 'val', 'test'])

	df.to_csv(filename)
	print()

class Generic_WSI_Classification_Dataset(Dataset):
	def __init__(self,
		csv_path = 'dataset_csv/ccrcc_clean.csv',
		shuffle = False, 
		seed = 7, 
		print_info = True,
		label_dict = {},
		filter_dict = {},
		ignore=[],
		patient_strat=False, # 是否按照病人进行划分
		label_col = None,
		patient_voting = 'max',
		):
		"""
		Args:
			csv_file (string): Path to the csv file with annotations.
			shuffle (boolean): Whether to shuffle
			seed (int): random seed for shuffling the data
			print_info (boolean): Whether to print a summary of the dataset
			label_dict (dict): Dictionary with key, value pairs for converting str labels to int
			ignore (list): List containing class labels to ignore
		"""
		self.label_dict = label_dict
		self.num_classes = len(set(self.label_dict.values()))
		self.seed = seed
		self.print_info = print_info
		self.patient_strat = patient_strat
		self.train_ids, self.val_ids, self.test_ids  = (None, None, None)
		self.data_dir = None
		if not label_col:
			label_col = 'label'
		self.label_col = label_col

		slide_data = pd.read_csv(csv_path)
		# print("csv_path",csv_path)
		# print("slide_data",slide_data)
		slide_data = self.filter_df(slide_data, filter_dict) # 根据给定条件过滤数据（如只保留某些类型）
		slide_data = self.df_prep(slide_data, self.label_dict, ignore, self.label_col) # 标签转换

		###shuffle data
		if shuffle:
			np.random.seed(seed)
			np.random.shuffle(slide_data)

		self.slide_data = slide_data

		if patient_strat==True:
			self.patient_data_prep(patient_voting) # 是否按病人聚合标签
		self.cls_ids_prep() # 为每一类记录其 slide / 病人 索引

		if print_info:
			self.summarize()

	# 按照类别label，为每一类分别记录属于该类的索引列表（IDs）
	def cls_ids_prep(self):
		# store ids corresponding each class at the patient or case level # 统计病人（case_id）级别每类的索引
		if self.patient_strat==True:
			self.patient_cls_ids = [[] for i in range(self.num_classes)] # 创建一个长度为类别数的空列表，用于记录每类病人的索引		
			for i in range(self.num_classes):
				self.patient_cls_ids[i] = np.where(self.patient_data['label'] == i)[0] 
    
		# store ids corresponding each class at the slide level # 统计 slide 级别每类的索引
		self.slide_cls_ids = [[] for i in range(self.num_classes)]
		for i in range(self.num_classes):
			self.slide_cls_ids[i] = np.where(self.slide_data['label'] == i)[0]


	# 根据同一个 patient（病人）下的多个 slide，生成一个对应的 patient-level 标签（label）
	def patient_data_prep(self, patient_voting='max'):
		patients = np.unique(np.array(self.slide_data['case_id'])) # get unique patients
		patient_labels = []
		
		for p in patients:
			locations = self.slide_data[self.slide_data['case_id'] == p].index.tolist()
			assert len(locations) > 0
			label = self.slide_data['label'][locations].values
			if patient_voting == 'max':
				label = label.max() # get patient label (MIL convention)
			elif patient_voting == 'maj':
				label = stats.mode(label)[0]
			else:
				raise NotImplementedError
			patient_labels.append(label)
		
		self.patient_data = {'case_id':patients, 'label':np.array(patient_labels)}

	@staticmethod
	# 根据ignore过滤数据
	def df_prep(data, label_dict, ignore, label_col):
		if label_col != 'label':
			data['label'] = data[label_col].copy()

		mask = data['label'].isin(ignore) 
		data = data[~mask] # 去掉label在ignore中的样本
		data.reset_index(drop=True, inplace=True) # 重置 DataFrame 索引
		for i in data.index:
			key = data.loc[i, 'label']
			data.at[i, 'label'] = label_dict[key] # 把字符串标签（如 "LUAD", "LUSC"）转换为数字标签（如 0, 1）
		return data

	# 根据filter_dict过滤数据
	def filter_df(self, df, filter_dict={}):
		if len(filter_dict) > 0: # 如果过滤条件字典非空，才进行过滤
			filter_mask = np.full(len(df), True, bool) # 初始化一个全是True的掩码，表示所有行一开始都保留
			# assert 'label' not in filter_dict.keys()
			for key, val in filter_dict.items():
				mask = df[key].isin(val)
				filter_mask = np.logical_and(filter_mask, mask)
			df = df[filter_mask]
		return df

	def __len__(self):
		if self.patient_strat:
			return len(self.patient_data['case_id'])

		else:
			return len(self.slide_data)

	def summarize(self):
		print("label column: {}".format(self.label_col))
		print("label dictionary: {}".format(self.label_dict))
		print("number of classes: {}".format(self.num_classes))
		print("slide-level counts: ", '\n', self.slide_data['label'].value_counts(sort = False))
		for i in range(self.num_classes):
			if self.patient_strat==True:
				print('Patient-LVL; Number of samples registered in class %d: %d' % (i, self.patient_cls_ids[i].shape[0]))
			print('Slide-LVL; Number of samples registered in class %d: %d' % (i, self.slide_cls_ids[i].shape[0]))

	def create_splits(self, k = 3, val_num = (25, 25), test_num = (40, 40), label_frac = 1.0, custom_test_ids = None):
		settings = {
					'n_splits' : k, # 要创建多少个fold
					'val_num' : val_num, 
					'test_num': test_num,
					'label_frac': label_frac,
					'seed': self.seed,
					'custom_test_ids': custom_test_ids # 自定义的测试样本索引（如指定某些slide为test）
					}

		if self.patient_strat: # 如果以患者为单位 stratify（病人为基本单位）
			settings.update({'cls_ids' : self.patient_cls_ids, 'samples': len(self.patient_data['case_id'])})
		else:
			settings.update({'cls_ids' : self.slide_cls_ids, 'samples': len(self.slide_data)})

		self.split_gen = generate_split(**settings)

	# 从 self.split_gen 生成器中获取一组划分（train/val/test 的索引），并根据是否是 patient-level stratify 来设置最终的 train_ids、val_ids、test_ids
	def set_splits(self,start_from=None):
		if start_from:
			ids = nth(self.split_gen, start_from)

		else:
			ids = next(self.split_gen)

		if self.patient_strat:
			slide_ids = [[] for i in range(len(ids))] 

			for split in range(len(ids)): # 对应 train，val或者test 
				for idx in ids[split]: 
					case_id = self.patient_data['case_id'][idx] # 病人的ID
					slide_indices = self.slide_data[self.slide_data['case_id'] == case_id].index.tolist() # 找出这个病人对应的所有 slide（可能有多个），返回其在 self.slide_data 中的索引
					slide_ids[split].extend(slide_indices) # 将这些 slide 的索引加入当前划分（train/val/test）中

			self.train_ids, self.val_ids, self.test_ids = slide_ids[0], slide_ids[1], slide_ids[2]

		else:
			self.train_ids, self.val_ids, self.test_ids = ids # 直接使用原始 slide 索引

	def get_split_from_df(self, all_splits, use_h5=None, h5_folder_name=None, split_key='train'):
		split = all_splits[split_key]
		split = split.dropna().reset_index(drop=True)
  
		# print("self.slide_data['slide_id2']",self.slide_data['slide_id2'])
		# print("split.tolist()",split.tolist())

		if len(split) > 0: #########
			mask = self.slide_data['slide_id2'].isin(split.tolist())
			# mask = self.slide_data['slide_id'].isin(split.tolist()) # 在 self.slide_data 中找出所有 slide_id 属于 split 列表的行
			# print("mask", mask)
			df_slice = self.slide_data[mask].reset_index(drop=True)
			split = Generic_Split(df_slice, use_h5=use_h5, h5_folder_name = h5_folder_name, data_dir=self.data_dir, num_classes=self.num_classes)
		else:
			print("split = None")
			split = None
		
		return split

	def get_merged_split_from_df(self, all_splits, split_keys=['train']):
		merged_split = []
		for split_key in split_keys:
			split = all_splits[split_key]
			split = split.dropna().reset_index(drop=True).tolist()
			merged_split.extend(split)

		if len(split) > 0:
			mask = self.slide_data['slide_id'].isin(merged_split)
			df_slice = self.slide_data[mask].reset_index(drop=True)
			split = Generic_Split(df_slice, use_h5=self.use_h5, h5_folder_name = self.h5_folder_name, data_dir=self.data_dir, num_classes=self.num_classes)
		else:
			split = None
		
		return split


	def return_splits(self, from_id=True, use_h5=None, h5_folder_name=None, csv_path=None):
		# 删掉slide_id列，然后把slide_id2列名变为slide_id
		# print("!!!!self.slide_data",self.slide_data)
		slide_data = self.slide_data.drop(columns=["slide_id"])
		slide_data = slide_data.rename(columns={"slide_id2": "slide_id"})
		# print("after!!!!self.slide_data",slide_data)
		# print("self.train_ids",self.train_ids)

		if from_id:
			if len(self.train_ids) > 0:
				train_data = slide_data.loc[self.train_ids].reset_index(drop=True)
				# print("train_data",train_data)
				train_split = Generic_Split(train_data, use_h5=use_h5, h5_folder_name = h5_folder_name, data_dir=self.data_dir, num_classes=self.num_classes)

			else:
				train_split = None
			
			if len(self.val_ids) > 0:
				val_data = slide_data.loc[self.val_ids].reset_index(drop=True)
				val_split = Generic_Split(val_data, use_h5=use_h5, h5_folder_name = h5_folder_name, data_dir=self.data_dir, num_classes=self.num_classes)

			else:
				val_split = None
			
			if len(self.test_ids) > 0:
				test_data = slide_data.loc[self.test_ids].reset_index(drop=True)
				test_split = Generic_Split(test_data, use_h5=use_h5, h5_folder_name = h5_folder_name, data_dir=self.data_dir, num_classes=self.num_classes)
			
			else:
				test_split = None
			
		
		else: ######
			print("not from_id")
			assert csv_path 
   			# all_splits = pd.read_csv(csv_path, dtype=self.slide_data['slide_id'].dtype)
			all_splits = pd.read_csv(csv_path, dtype=self.slide_data['slide_id2'].dtype)  # Without "dtype=self.slide_data['slide_id'].dtype", read_csv() will convert all-number columns to a numerical type. Even if we convert numerical columns back to objects later, we may lose zero-padding in the process; the columns must be correctly read in from the get-go. When we compare the individual train/val/test columns to self.slide_data['slide_id'] in the get_split_from_df() method, we cannot compare objects (strings) to numbers or even to incorrectly zero-padded objects/strings. An example of this breaking is shown in https://github.com/andrew-weisman/clam_analysis/tree/main/datatype_comparison_bug-2021-12-01.
			print("return_splits use_h5",use_h5,h5_folder_name)
			train_split = self.get_split_from_df(all_splits, use_h5, h5_folder_name, 'train')
			val_split = self.get_split_from_df(all_splits, use_h5, h5_folder_name, 'val')
			test_split = self.get_split_from_df(all_splits, use_h5, h5_folder_name, 'test')
			
		return train_split, val_split, test_split

	def get_list(self, ids):
		return self.slide_data['slide_id'][ids]

	def getlabel(self, ids):
		return self.slide_data['label'][ids]

	def __getitem__(self, idx):
		return None

	def test_split_gen(self, return_descriptor=False):

		if return_descriptor:
			index = [list(self.label_dict.keys())[list(self.label_dict.values()).index(i)] for i in range(self.num_classes)]
			columns = ['train', 'val', 'test']
			df = pd.DataFrame(np.full((len(index), len(columns)), 0, dtype=np.int32), index= index,
							columns= columns)

		count = len(self.train_ids)
		print('\nnumber of training samples: {}'.format(count))
		labels = self.getlabel(self.train_ids)
		unique, counts = np.unique(labels, return_counts=True)
		for u in range(len(unique)):
			print('number of samples in cls {}: {}'.format(unique[u], counts[u]))
			if return_descriptor:
				df.loc[index[u], 'train'] = counts[u]
		
		count = len(self.val_ids)
		print('\nnumber of val samples: {}'.format(count))
		labels = self.getlabel(self.val_ids)
		unique, counts = np.unique(labels, return_counts=True)
		for u in range(len(unique)):
			print('number of samples in cls {}: {}'.format(unique[u], counts[u]))
			if return_descriptor:
				df.loc[index[u], 'val'] = counts[u]

		count = len(self.test_ids)
		print('\nnumber of test samples: {}'.format(count))
		labels = self.getlabel(self.test_ids)
		unique, counts = np.unique(labels, return_counts=True)
		for u in range(len(unique)):
			print('number of samples in cls {}: {}'.format(unique[u], counts[u]))
			if return_descriptor:
				df.loc[index[u], 'test'] = counts[u]

		assert len(np.intersect1d(self.train_ids, self.test_ids)) == 0
		assert len(np.intersect1d(self.train_ids, self.val_ids)) == 0
		assert len(np.intersect1d(self.val_ids, self.test_ids)) == 0

		if return_descriptor:
			return df

	def save_split(self, filename):
		train_split = self.get_list(self.train_ids)
		val_split = self.get_list(self.val_ids)
		test_split = self.get_list(self.test_ids)
		df_tr = pd.DataFrame({'train': train_split})
		df_v = pd.DataFrame({'val': val_split})
		df_t = pd.DataFrame({'test': test_split})
		df = pd.concat([df_tr, df_v, df_t], axis=1) 
		df.to_csv(filename, index = False)


class Generic_MIL_Dataset(Generic_WSI_Classification_Dataset):
	def __init__(self,
		data_dir, 
		h5_folder_name,
		use_h5,
		**kwargs):
	
		super(Generic_MIL_Dataset, self).__init__(**kwargs)
		self.data_dir = data_dir
		self.use_h5 = use_h5 # 是否使用.h5文件作为输入的开关，默认使用.pt
		self.h5_folder_name = h5_folder_name

	# def load_from_h5(self, toggle):
	# 	self.use_h5 = toggle

	def __getitem__(self, idx): # 返回第 idx 个样本的特征和标签
		slide_id = self.slide_data['slide_id2'][idx]
		label = self.slide_data['label'][idx]
		if type(self.data_dir) == dict:
			source = self.slide_data['source'][idx]
			data_dir = self.data_dir[source]
		else:
			data_dir = self.data_dir
		
		# if not self.use_h5: # ** pt files **
		# 	if self.data_dir:
		# 		full_path = os.path.join(data_dir, 'pt_files', '{}.pt'.format(slide_id))
		# 		features = torch.load(full_path)
		# 		return features, label
			
		# 	else:
		# 		return slide_id, label

		# else: # ** h5 files **
			# if self.h5_folder_name == 'h5_files' or self.h5_folder_name == 'h5_files_annotations':
			# 	full_path = os.path.join(data_dir,self.h5_folder_name,'{}.h5'.format(slide_id))
			# 	with h5py.File(full_path,'r') as hdf5_file:
			# 		features = hdf5_file['features'][:]
			# 		coords = hdf5_file['coords'][:]
			# 	features = torch.from_numpy(features)
			# 	return features, label, coords

		if self.h5_folder_name == 'h5_files_labels':
			full_path = os.path.join(data_dir,'h5_files_labels','{}.h5'.format(slide_id))
			if not os.path.exists(full_path):
				full_path = os.path.join(data_dir,'h5_files_labels','{}_tiff.h5'.format(slide_id))
	
			hdf5_file = safe_open_h5(full_path)

			# with h5py.File(full_path,'r') as hdf5_file:
			features = hdf5_file['features'][:]
			coords = hdf5_file['coords'][:]
			labels_mask = hdf5_file['labels'][:]
			slide_id2 = slide_id

			features = torch.from_numpy(features)
			return features, label, coords, labels_mask, slide_id2 # -> utils_clam/utils.py -> collate_MIL

	def get_data_by_slide_id(self, slide_id2):
		# 在 slide_data 中查找对应索引
		matches = self.slide_data[self.slide_data['slide_id2'] == slide_id2]
		if len(matches) == 0:
			raise ValueError(f"Slide ID {slide_id2} not found in dataset.")

		idx = matches.index[0]
		label = self.slide_data.loc[idx, 'label']

		if isinstance(self.data_dir, dict):
			source = self.slide_data.loc[idx, 'source']
			data_dir = self.data_dir[source]
		else:
			data_dir = self.data_dir

		# if not self.use_h5:
		# 	full_path = os.path.join(data_dir, 'pt_files', f'{slide_id2}.pt')
		# 	features = torch.load(full_path)
		# 	return features, label

		# if self.h5_folder_name in ['h5_files', 'h5_files_annotations']:
		# 	full_path = os.path.join(data_dir, self.h5_folder_name, f'{slide_id2}.h5')
		# 	with h5py.File(full_path, 'r') as hdf5_file:
		# 		features = hdf5_file['features'][:]
		# 		coords = hdf5_file['coords'][:]
		# 	features = torch.from_numpy(features)
		# 	batch = {
		# 		'features': features, 
		# 		'label': label, 
		# 		'coords': coords,
		# 	}
		# 	return batch

		if self.h5_folder_name == 'h5_files_labels':
			full_path = os.path.join(data_dir, 'h5_files_labels', f'{slide_id2}.h5')
			if not os.path.exists(full_path):
				full_path = os.path.join(data_dir, 'h5_files_labels', f'{slide_id2}_tiff.h5')
			hdf5_file = safe_open_h5(full_path)
			features = hdf5_file['features'][:]
			coords = hdf5_file['coords'][:]
			bag_size = coords.shape[0]
			labels_mask = hdf5_file['labels'][:]
			features = torch.from_numpy(features)
			batch = {
				'features': features, 
				'label': label, 
				'coords': coords,
				'labels_mask': labels_mask,
				'slide_id2': slide_id2,
				'bag_size': bag_size,
			}
			return batch


class Generic_Split(Generic_MIL_Dataset):
	def __init__(self, slide_data, use_h5, h5_folder_name, data_dir=None, num_classes=2):
		self.use_h5 = use_h5 
		self.h5_folder_name = h5_folder_name
		print("Generic_Split",self.use_h5,self.h5_folder_name)
		self.slide_data = slide_data
		self.data_dir = data_dir
		self.num_classes = num_classes
		self.slide_cls_ids = [[] for i in range(self.num_classes)]
		for i in range(self.num_classes):
			self.slide_cls_ids[i] = np.where(self.slide_data['label'] == i)[0]

	def __len__(self):
		return len(self.slide_data)
		


