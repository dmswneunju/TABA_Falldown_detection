import os
import pandas as pd

# CSV 파일이 포함된 디렉토리 설정
csv_dir = '/media/leeej/My Passport/시니어 이상행동 영상/Training/output_frames'

def find_missing_values_in_csv_files(directory):
    # 총 데이터 개수 및 결측치 개수를 저장할 변수 초기화
    total_data_count = 0
    total_missing_values_count = 0
    total_columns = {}
    total_missing_per_column = {}

    # 디렉토리 내 모든 CSV 파일 찾기
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.csv'):
                csv_path = os.path.join(root, file)
                print(f"Processing file {csv_path}...")

                # CSV 파일 읽기
                df = pd.read_csv(csv_path)

                # 총 데이터 개수 계산 (전체 셀의 수)
                data_count = df.size
                total_data_count += data_count

                # 결측치 개수 계산
                missing_values_count = df.isnull().sum().sum()
                total_missing_values_count += missing_values_count

                # 개별 파일의 결측치 정보 출력
                if missing_values_count > 0:
                    print(f"\nMissing Values in {csv_path}:")
                    print(df[df.isnull().any(axis=1)])
                    print(f"\nTotal data count in this file: {data_count}")
                    print(f"Missing values count in this file: {missing_values_count}")

                # 각 열별 결측치 개수 계산
                for column in df.columns:
                    if column not in total_columns:
                        total_columns[column] = 0
                        total_missing_per_column[column] = 0

                    total_columns[column] += df[column].size
                    total_missing_per_column[column] += df[column].isnull().sum()

    return total_data_count, total_missing_values_count, total_columns, total_missing_per_column

# 결측치 찾기 및 총 데이터 개수, 총 결측치 개수 저장
total_data_count, total_missing_values_count, total_columns, total_missing_per_column = find_missing_values_in_csv_files(csv_dir)

# 결측치 비율 계산
missing_values_percentage = (total_missing_values_count / total_data_count) * 100 if total_data_count > 0 else 0

# 총 데이터 개수, 결측치 개수 및 결측치 비율 출력
print(f"\nTotal Data Count in all CSV files: {total_data_count}")
print(f"Total Missing Values Count in all CSV files: {total_missing_values_count}")
print(f"Missing Values Percentage: {missing_values_percentage:.2f}%")

# 각 열별 결측치 비율 계산 및 출력
print("\nColumn-wise Missing Values Percentage:")
for column in total_columns:
    total = total_columns[column]
    missing = total_missing_per_column[column]
    percentage = (missing / total) * 100 if total > 0 else 0
    print(f"Column: {column}, Total: {total}, Missing: {missing}, Missing Percentage: {percentage:.2f}%")
