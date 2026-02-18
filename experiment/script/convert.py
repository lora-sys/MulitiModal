import os
import re
import pandas as pd
import numpy as np
import logging
from tqdm import tqdm


logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s')

class DataProcessor :
    def __init__(self,base_path,save_path,sample_len=1000):
        self.base_path= base_path
        self.save_path =  save_path
        self.sample_len = sample_len
        self.label_map = {
            "身体表征很差": 0, 
            "身体表征一般": 1, 
            "身体表征正常": 2, 
            "身体表征良好": 3
        }
        # 定义填充值，传感器无数据填0
        self.pad_value=0.0
    
    def _pad_or_truncate(self,sequence):
        """
        生产级核心逻辑：确保序列长度严格一致
        截断过长，填充过短
        """
        seq_len = len(sequence)
        if seq_len >= self.sample_len :
            return sequence[:self.sample_len]
        else :
        # 使用 np.pad 进行填充 (前面填充或后面填充根据业务定，这里示范后面填充)
            pad_width = self.sample_len - seq_len
            return np.pad(sequence, (0, pad_width), 'constant', constant_values=self.pad_value)
        
        
    def process(self):
        dynamic_list = []
        static_list = []
        label_list = []
        logging.info(f"🚀 开始处理数据，根目录: {self.base_path}")
        
        for folder_name, lebel_val in self.label_map.items():
            folder_path = os.path.join(self.base_path,folder_name)
            if not os.path.exists(folder_path):
                logging.warning(f"文件不存在:{folder_path}")
            
            
            files = [f for f in os.listdir(folder_path) if f.endswith(".csv")  ]
            
            for file in tqdm(files, desc=f"processing {folder_name}"):
                try :
                    # 1. 静态特征提取
                    numbers = re.findall(r"\d+",file)
                    if len(numbers) < 5 :
                        logging.warning(f"文件名格式错误，跳过:{file}")
                        continue
                    
                    static_feat = np.array([float(x) for x in numbers[1 : 5]],dtype=np.float32)
                    
                    # 2. 动态特征提取
                    
                    df = pd.read_csv(os.path.join(folder_path,file))
                    
                    # 校验列是否存在
                    if '压力传感器1' not in df.columns or '压力传感器2' not in df.columns:
                        logging.warning(f"CSV缺少必要的列，跳过: {file}")
                        continue
                    
                    # 提取强制对齐长度
                    p1 = self._pad_or_truncate(df['压力传感器1'].values)
                    p2 = self._pad_or_truncate(df['压力传感器2'].values) 
                    
                    # 组装数据
                    # 形状(2,1000) - > channel first                                        
                    dynamic_feat = np.vstack([p1,p2]).astype(np.float32)
                    
                    dynamic_list.append(dynamic_feat)
                    static_list.append(static_feat)
                    label_list.append(lebel_val)
                
                except Exception as e :
                    logging.error(f"处理文件{file} 出错 {e}") 
                    continue
        
        # 转换为numpy数组
        X_dynamic = np.array(dynamic_list,dtype=np.float32)  # (n,2,1000)
        X_static = np.array(static_list,dtype=np.float32)   # (n,4)
        Y= np.array(label_list,dtype=np.int64)
        
        # 保存为 npz 
        
        np.savez_compressed(
            self.save_path,
            X_dynamic = X_dynamic,
            X_static = X_static,
            Y=Y
        )                        
        logging.info(f"✅ 处理完成！数据已保存至: {self.save_path}")
        logging.info(f"📊 动态张量形状: {X_dynamic.shape}")
        logging.info(f"📊 静态张量形状: {X_static.shape}")
        logging.info(f"📊 标签分布: {np.bincount(Y)}")



if __name__ == "__main__":
    processor = DataProcessor(
        base_path="experiment/data",
        save_path="experiment/model/processed_data.npz", # 推荐使用 .npz 格式
        sample_len=1000
    )
    processor.process()