import pandas as pd

# 최종 DataFrame 로드 (이미 CSV 파일로 저장된 경우)
output_csv_path = '/media/leeej/My Passport/시니어 이상행동 영상/Training/final_dataframe.csv'
df = pd.read_csv(output_csv_path)

# 결측치 확인
print("DataFrame with Missing Values:")
print(df)

# 결측치가 있는 행 찾기
missing_values = df[df.isnull().any(axis=1)]
print("\nRows with Missing Values:")
print(missing_values)

# 결측치 개수 확인
missing_values_count = df.isnull().sum().sum()
total_data_count = df.size
missing_values_percentage = (missing_values_count / total_data_count) * 100 if total_data_count > 0 else 0

print("\nTotal Data Count in the DataFrame: ", total_data_count)
print("Total Missing Values Count in the DataFrame: ", missing_values_count)
print(f"Missing Values Percentage: {missing_values_percentage:.2f}%")

# 각 열별 결측치 비율 계산
print("\nColumn-wise Missing Values Percentage:")
for column in df.columns:
    total = df[column].size
    missing = df[column].isnull().sum()
    percentage = (missing / total) * 100 if total > 0 else 0
    print(f"Column: {column}, Total: {total}, Missing: {missing}, Missing Percentage: {percentage:.2f}%")
