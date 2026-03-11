import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
# ==================== 配置 ====================
ROW_DIR = os.path.join(os.path.dirname(__file__), "row_data")
CLEAN_DIR=os.path.join(os.path.dirname(__file__), "clean_data")
os.makedirs(CLEAN_DIR, exist_ok=True)
# ==================== 1. 体重秤 Cleaner ====================
def clean_scale_data():
    df = pd.read_csv(os.path.join(ROW_DIR, "体重秤.csv"))
    
    # 统一命名
    df = df.rename(columns={
        '年龄': 'age',
        '0男 1女': 'gender',
        'bmi': 'bmi',
        '脂肪率%': 'fat_rate',
        '肌肉量KG': 'muscle_mass',
        '内脏脂肪等级': 'visceral_fat',
        '基础代谢率': 'bmr',
        '身体得分': 'body_score'  # 原始分数
    })
    
    # 计算BMI（如果缺失）
    df['bmi_calculated'] = df['体重KG'] / ((df['身高'] / 100) ** 2)
    df['bmi'] = df['bmi'].fillna(df['bmi_calculated'])
    
    # 选择输出列（去除 user_id、身高、体重、测量时间）
    output_cols = ['age', 'gender', 'bmi', 'fat_rate', 'muscle_mass', 'visceral_fat', 'bmr', 'body_score']
    
    df[output_cols].to_csv(os.path.join(CLEAN_DIR, "scale_clean.csv"), index=False)
    print(f"✅ 体重秤清洗完成: {len(df)} 行, {len(output_cols)} 列")
# ==================== 2. 新建工作表 Cleaner ====================
def clean_member_data():
    df = pd.read_csv(os.path.join(ROW_DIR, "新建 XLSX 工作表(4).csv"))
    
    # 统一命名
    df = df.rename(columns={
        '性别0男1女': 'gender',
        '心率': 'heart_rate',
        '微循环': 'microcirculation',
        '血氧': 'blood_oxygen',
        '疲劳指数': 'fatigue_index'  # 原始分数
    })
    
    # 计算年龄
    def calculate_age(birth_date):
        try:
            birth = datetime.strptime(str(birth_date), "%m/%d/%Y")
            today = datetime.now()
            return today.year - birth.year - ((today.month, today.day) < (birth.month, birth.day))
        except:
            return np.nan
    
    df['age'] = df['出生日期'].apply(calculate_age)
    
    # 计算BMI
    df['bmi'] = df['体重'] / ((df['身高'] / 100) ** 2)
    
    # 选择输出列
    output_cols = ['age', 'gender', 'bmi', 'heart_rate', 'microcirculation', 'blood_oxygen', 'fatigue_index']
    
    df[output_cols].to_csv(os.path.join(CLEAN_DIR, "member_clean.csv"), index=False)
    print(f"✅ 新建工作表清洗完成: {len(df)} 行, {len(output_cols)} 列")
# ==================== 3. 舌面诊断 Cleaner ====================
def clean_tongue_data():
    """清洗舌面诊断数据（修复版）"""
    df = pd.read_csv(os.path.join(ROW_DIR, "舌面诊断.csv"), header=None)
    df = df.rename(columns={0: 'user_id', 1: 'json_str'})
    
    def parse_tongue_json(json_str):
        try:
            data = json.loads(json_str)
            result = {}
            
            # 1. BMI（从bmiEvaluation中提取）
            bmi_data = data.get('result', {}).get('bmiEvaluation', {}).get('bmi', {})
            result['bmi'] = bmi_data.get('value', np.nan)
            
            # 2. 健康指数（顶层字段）
            result['health_index'] = data.get('result', {}).get('healthIndex', np.nan)
            
            # result['constitution'] = data.get('result', {}).get('constitutionNames', '')
        
            
            # 4. 面部特征
            face = data.get('result', {}).get('characterMap', {}).get('face', {})
            result['left_eye'] = face.get('heiyanquanLeft', {}).get('name', '')
            result['right_eye'] = face.get('heiyanquanRight', {}).get('name', '')
            result['face_color'] = face.get('mianse', {}).get('name', '')
            
            # 5. 舌苔特征（修复路径）
            tongue = data.get('result', {}).get('tongue', {})
            color_of_tongue = tongue.get('colorOfTongue', {})
            result['tongue_color'] = color_of_tongue.get('name', '')
            
            return pd.Series(result)
        except Exception as e:
            print(f"JSON解析错误: {e}")
            return pd.Series({
                'bmi': np.nan, 'health_index': np.nan, 
                'left_eye': '', 'right_eye': '', 'face_color': '', 'tongue_color': ''
            })
    
    # 应用解析函数
    parsed = df['json_str'].apply(parse_tongue_json)
    
    # 统计空值
    print("\n空值统计:")
    for col in parsed.columns:
        empty_count = parsed[col].isna().sum() + (parsed[col] == '').sum()
        print(f"  {col}: {empty_count}/{len(parsed)} ({empty_count/len(parsed)*100:.1f}%)")
    
    # 保存
    parsed.to_csv(os.path.join(CLEAN_DIR, "tongue_clean.csv"), index=False)
    print(f"\n✅ 舌面诊断清洗完成: {len(parsed)} 行, {len(parsed.columns)} 列")
# ==================== 主函数 ====================
def main():
    print("=" * 50)
    print("开始清洗数据...")
    print("=" * 50)
    
    clean_scale_data()
    clean_member_data()
    clean_tongue_data()
    
    print("=" * 50)
    print("清洗完成！")
    print("=" * 50)
    
    print("\n输出文件:")
    for file in os.listdir(CLEAN_DIR):
        filepath = os.path.join(CLEAN_DIR, file)
        if os.path.isfile(filepath):
            df = pd.read_csv(filepath)
            print(f"  - {file}: {len(df)} 行, {len(df.columns)} 列")
if __name__ == "__main__":
    main()


