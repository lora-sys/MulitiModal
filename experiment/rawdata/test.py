import pandas as pd 



if __name__ == "__main__":
    print("---")
    df = pd.read_csv("experiment/rawdata/row_data/新建 XLSX 工作表(4).csv")
    print(df.columns.tolist())
    print("---")  
    df1 =pd.read_csv("experiment/rawdata/row_data/体重秤.csv")
    print(df1.columns.tolist())
    print("---")
    df2 =pd.read_csv("experiment/rawdata/row_data/舌面诊断.csv")
    print(df2.columns.tolist())
    print("---")
    # print(df2.head(2))
    

