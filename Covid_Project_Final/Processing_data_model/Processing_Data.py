import pandas as pd

# Đọc dữ liệu
df = pd.read_csv(r"C:\Users\ming2\OneDrive\Documents\FPT_University\Semester 4\DAP\Source_Code\compact.csv")

# Hiển thị thông tin cơ bản về dataset
print("-" * 50)
print("Đọc dữ liệu thành công. Dữ liệu có {} dòng và {} cột.".format(df.shape[0], df.shape[1]))
print("Thông tin cơ bản về dataset:")
print("-" * 50)
print(df.info())
print("\nMẫu dữ liệu:")
print("-" * 50)
print(df.head())



# Kiểm tra missing values
print("Số lượng giá trị thiếu trong mỗi cột:")
print("-" * 50)
print(df.isnull().sum())

# Tính phần trăm giá trị thiếu
missing_percentage = (df.isnull().sum() / len(df)) * 100
print("\nPhần trăm giá trị thiếu trong mỗi cột:")
print("-" * 50)
print(missing_percentage)

# Xử lý missing values
# Đối với stringency_index, sử dụng phương pháp forward fill theo giả sử rằng chính sách vẫn như cũ nếu không có thay đổi mới
df['stringency_index'] = df.groupby('country')['stringency_index'].ffill()
df['people_fully_vaccinated'] = df.groupby('country')['people_fully_vaccinated'].ffill()

df.fillna(0, inplace=True)

# Cắt dữ liệu chỉ lấy các bản ghi đến ngày 2025-04-27 trở về trước
df['date'] = pd.to_datetime(df['date'])  
print(df['date'].min(), df['date'].max())
cutoff_date = pd.to_datetime('2025-04-27')
df = df[df['date'] <= cutoff_date]
df["date"] = df["date"].dt.strftime("%d/%m/%Y")
print(f"Thời gian bắt đầu: {df['date'].min()}")
print(f"Thời gian kết thúc: {df['date'].max()}")

print(df.head(5))

print("Số lượng giá trị thiếu sau khi xử lý:")
print("-" * 50)
print(df.isnull().sum())
print(df.shape)

# Chọn các cột cần thiết
selected_columns = [
    'code', 'continent','country', 'continent', 'date',
    'total_cases', 'new_cases', 'total_deaths', 'new_deaths',
    'people_fully_vaccinated_per_hundred', 'new_vaccinations_smoothed',
    'population', 'stringency_index', 'people_fully_vaccinated'
]



# Lọc bỏ các tên nước không hợp lệ như các vùng lục địa, các lãnh thổ phụ thuộc, các nước quá nhỏ để tránh gây nhiễu dữ liệu
exclude_keywords = [
    "World", "income", "countries", "excl.", "European Union", "High-income", "Upper-middle", "Lower-middle", "Low-income", "Olympics",
    'Africa', 'Asia', 'Europe', 'North America', 'South America', 'Oceania',
    'England', 'Scotland', 'Wales', 'England & Wales', 'Northern Ireland',
    'American Samoa', 'Guam', 'Puerto Rico', 'United States Virgin Islands',
    'Northern Mariana Islands', 'Bermuda', 'Cayman Islands', 'Montserrat',
    'Anguilla', 'British Virgin Islands', 'Turks and Caicos Islands',
    'Greenland', 'Hong Kong', 'Macao', 'Taiwan',
    'Reunion', 'Mayotte', 'Guadeloupe', 'Martinique', 'French Guiana',
    'French Polynesia', 'Saint Martin (French part)', 'Saint Barthelemy',
    'Saint Pierre and Miquelon', 'New Caledonia', 'Wallis and Futuna',
    'Cook Islands', 'Niue', 'Tokelau', 'Pitcairn',
    'Bonaire Sint Eustatius and Saba', 'Curacao', 'Aruba',
    'Kosovo', 'Transnistria', 'Vatican',
    'Isle of Man', 'Jersey', 'Guernsey', 'Saint Helena'
]
country_list = [c for c in df["country"].unique().tolist() if not any(kw in c for kw in exclude_keywords)]
df_country = df[df["country"].isin(country_list)]

# Lọc dataset với các cột đã chọn
df_cleaned = df_country[selected_columns].copy()
df_cleaned = df_cleaned.rename(columns={
    "country": "location",
    "code": "iso_code",
})
print(df_cleaned.head(2))
# Lưu dữ liệu đã xử lý vào file CSV mới
df.to_csv("Covid19_cleaned.csv", index=False)
df_cleaned.to_csv("Covid19_cleaned_to_model_2.csv", index=False)
print("\nThông tin về dữ liệu đã làm sạch:")
print("-" * 50)
print(f"Số lượng bản ghi: {len(df_cleaned)}")
print(f"Số lượng quốc gia: {df_cleaned['location'].nunique()}")
print(f"Số lượng châu lục: {df_cleaned['continent'].nunique()}")
print(f"Số lượng châu lục: {df_cleaned['continent'].nunique()}")
