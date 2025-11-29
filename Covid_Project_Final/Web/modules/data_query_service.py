import pandas as pd
from pathlib import Path
from datetime import datetime
import numpy as np

class DataQueryService:
    def __init__(self, data_path=None):
        self.data = None
        self.load_data(data_path)

    def load_data(self, data_path=None):
        if data_path is None:
            data_path = Path(__file__).parent.parent / "data" / "Covid19_cleaned_to_model.csv"
        try:
            self.data = pd.read_csv(data_path)
            # Cải thiện việc xử lý ngày tháng
            self.data["date"] = pd.to_datetime(self.data["date"], errors='coerce')
            # Loại bỏ các dòng có ngày không hợp lệ
            self.data = self.data.dropna(subset=['date'])
            # Chuẩn hóa timezone về UTC
            if self.data["date"].dt.tz is not None:
                self.data["date"] = self.data["date"].dt.tz_convert('UTC').dt.tz_localize(None)
            
            print(f"Đã tải {len(self.data)} bản ghi.")
            print(f"Khoảng thời gian dữ liệu: {self.data['date'].min()} đến {self.data['date'].max()}")
            print(self.data.head(10))
            print("Dữ liệu COVID đã được tải vào DataQueryService.")
        except Exception as e:
            print(f"Lỗi khi tải dữ liệu vào DataQueryService: {e}")

    def _normalize_date(self, date_input):
        """Chuẩn hóa ngày đầu vào thành datetime object"""
        if isinstance(date_input, str):
            try:
                # Thử các format phổ biến
                formats = ['%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y', '%Y/%m/%d']
                for fmt in formats:
                    try:
                        return pd.to_datetime(date_input, format=fmt)
                    except ValueError:
                        continue
                # Nếu không match format nào, dùng pd.to_datetime tự động
                return pd.to_datetime(date_input)
            except:
                return None
        elif isinstance(date_input, datetime):
            return pd.to_datetime(date_input)
        else:
            return pd.to_datetime(date_input)

    def get_latest_data_date(self, country=None):
        if self.data is None or self.data.empty:
            return None
        try:
            if country:
                country_data = self.data[self.data["location"].str.lower() == country.lower()]
                if not country_data.empty:
                    return country_data["date"].max().date()
                else:
                    return None
            return self.data["date"].max().date()
        except Exception as e:
            print(f"Lỗi khi lấy ngày mới nhất: {e}")
            return None

    def get_overview_data(self):
        if self.data is None or self.data.empty:
            return "Không có dữ liệu để hiển thị tổng quan."
        
        try:
            latest_date = self.data["date"].max()
            latest_data = self.data[self.data['date'] == latest_date]
            
            # Xử lý NaN values
            total_cases = latest_data["total_cases"].fillna(0).sum()
            total_deaths = latest_data["total_deaths"].fillna(0).sum()
            total_vaccinated = latest_data["people_fully_vaccinated"].fillna(0).sum()
            
            response = f"Dữ liệu COVID-19 tổng quan đến ngày {latest_date.strftime('%d/%m/%Y')}:\n"
            response += f"- Tổng số ca nhiễm: {total_cases:,.0f}\n"
            response += f"- Tổng số ca tử vong: {total_deaths:,.0f}\n"
            response += f"- Tổng số người được tiêm chủng đầy đủ: {total_vaccinated:,.0f}\n"
            response += f"- Số quốc gia/vùng lãnh thổ có dữ liệu: {len(latest_data)}\n"
            
            return response
        except Exception as e:
            return f"Lỗi khi lấy dữ liệu tổng quan: {e}"

    def get_country_data_summary(self, country):
        if self.data is None or self.data.empty:
            return "Không có dữ liệu để hiển thị."
        
        try:
            # Tìm kiếm không phân biệt hoa thường
            country_data = self.data[self.data["location"].str.lower() == country.lower()]
            if country_data.empty:
                # Thử tìm kiếm gần đúng
                similar_countries = self.data[self.data["location"].str.contains(country, case=False, na=False)]
                if not similar_countries.empty:
                    available_countries = similar_countries["location"].unique()[:5]
                    return f"Không tìm thấy '{country}'. Có thể bạn muốn tìm: {', '.join(available_countries)}"
                return f"Không tìm thấy dữ liệu cho '{country}'."
            
            latest_date = country_data["date"].max()
            latest_country_data = country_data[country_data["date"] == latest_date].iloc[0]
            
            response = f"Dữ liệu COVID-19 mới nhất cho {latest_country_data['location']} đến ngày {latest_date.strftime('%d/%m/%Y')}:\n"
            response += f"- Tổng số ca nhiễm: {latest_country_data.get('total_cases', 0):,.0f}\n"
            response += f"- Ca nhiễm mới: {latest_country_data.get('new_cases', 0):,.0f}\n"
            response += f"- Tổng số ca tử vong: {latest_country_data.get('total_deaths', 0):,.0f}\n"
            response += f"- Ca tử vong mới: {latest_country_data.get('new_deaths', 0):,.0f}\n"
            
            # Xử lý dữ liệu vaccination có thể null
            vaccination_data = latest_country_data.get('people_fully_vaccinated', 0)
            if pd.isna(vaccination_data):
                response += f"- Số người được tiêm chủng đầy đủ: Chưa có dữ liệu\n"
            else:
                response += f"- Số người được tiêm chủng đầy đủ: {vaccination_data:,.0f}\n"
            
            return response
            
        except Exception as e:
            return f"Lỗi khi lấy dữ liệu cho {country}: {e}"

    def get_data_by_date_and_country(self, country, target_date):
        if self.data is None or self.data.empty:
            return "Không có dữ liệu để hiển thị."
        
        try:
            # Chuẩn hóa tên quốc gia
            country_data = self.data[self.data["location"].str.lower() == country.lower()]
            if country_data.empty:
                similar_countries = self.data[self.data["location"].str.contains(country, case=False, na=False)]
                if not similar_countries.empty:
                    available_countries = similar_countries["location"].unique()[:3]
                    return f"Không tìm thấy '{country}'. Có thể bạn muốn tìm: {', '.join(available_countries)}"
                return f"Không tìm thấy dữ liệu cho quốc gia '{country}'."

            # Chuẩn hóa ngày
            normalized_date = self._normalize_date(target_date)
            if normalized_date is None:
                return f"Định dạng ngày '{target_date}' không hợp lệ. Vui lòng sử dụng YYYY-MM-DD, DD/MM/YYYY hoặc MM/DD/YYYY."

            # Tìm dữ liệu cho ngày cụ thể
            specific_date_data = country_data[country_data["date"].dt.date == normalized_date.date()]
            
            if not specific_date_data.empty:
                data_row = specific_date_data.iloc[0]
                country_name = data_row['location']
                
                response = f"Dữ liệu COVID-19 cho {country_name} vào ngày {normalized_date.strftime('%d/%m/%Y')}:\n"
                response += f"- Tổng số ca nhiễm: {data_row.get('total_cases', 0):,.0f}\n"
                response += f"- Ca nhiễm mới: {data_row.get('new_cases', 0):,.0f}\n"
                response += f"- Tổng số ca tử vong: {data_row.get('total_deaths', 0):,.0f}\n"
                response += f"- Ca tử vong mới: {data_row.get('new_deaths', 0):,.0f}\n"
                
                vaccination_data = data_row.get('people_fully_vaccinated', 0)
                if pd.isna(vaccination_data):
                    response += f"- Số người được tiêm chủng đầy đủ: Chưa có dữ liệu\n"
                else:
                    response += f"- Số người được tiêm chủng đầy đủ: {vaccination_data:,.0f}\n"
                
                return response
            else:
                # Tìm ngày gần nhất có dữ liệu
                available_dates = country_data["date"].dt.date.unique()
                closest_date = min(available_dates, key=lambda x: abs((x - normalized_date.date()).days))
                
                return f"Không có dữ liệu cho {country} vào ngày {normalized_date.strftime('%d/%m/%Y')}.\n" \
                       f"Ngày gần nhất có dữ liệu: {closest_date.strftime('%d/%m/%Y')}"
                       
        except Exception as e:
            return f"Lỗi khi truy vấn dữ liệu: {e}"

    def get_available_countries(self):
        """Lấy danh sách các quốc gia có sẵn trong dữ liệu"""
        if self.data is None or self.data.empty:
            return []
        return sorted(self.data["location"].unique().tolist())

    def get_date_range_for_country(self, country):
        """Lấy khoảng thời gian có dữ liệu cho một quốc gia"""
        if self.data is None or self.data.empty:
            return None
        
        country_data = self.data[self.data["location"].str.lower() == country.lower()]
        if country_data.empty:
            return None
            
        return {
            'min_date': country_data["date"].min().date(),
            'max_date': country_data["date"].max().date(),
            'total_records': len(country_data)
        }

# Khởi tạo service toàn cục
def get_data_query_service():
    return DataQueryService()