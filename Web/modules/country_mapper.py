
# modules/country_mapper.py
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
import re

class CountryMapper:
    """Class để mapping quốc gia với model embedding một cách thông minh"""
    
    def __init__(self, data_path=None):
        self.data = None
        self.country_mapping = {}
        self.supported_countries = []
        self.country_aliases = {}
        self.label_encoder = LabelEncoder()
        self.load_data(data_path)
        self.setup_country_mapping()
        self.setup_aliases()
    
    def load_data(self, data_path=None):
        """Tải dữ liệu COVID"""
        if data_path is None:
            data_path = Path(__file__).parent.parent / "data" / "covid_cleaned_country_data.csv"
        
        try:
            self.data = pd.read_csv(data_path)
            self.data["date"] = pd.to_datetime(self.data["date"])
        except Exception as e:
            print(f"Lỗi khi tải dữ liệu: {e}")
    
    def setup_country_mapping(self):
        """Thiết lập mapping quốc gia dựa trên dữ liệu có sẵn"""
        if self.data is None:
            return
        
        # Lấy tất cả các quốc gia duy nhất từ cột 'location'
        all_countries = sorted(self.data["location"].unique().tolist())
        self.supported_countries = all_countries
        
        # Fit LabelEncoder với tất cả các quốc gia
        self.label_encoder.fit(self.supported_countries)
        
        # Tạo mapping từ tên quốc gia sang ID (chỉ số của LabelEncoder)
        self.country_mapping = {country: self.label_encoder.transform([country])[0] for country in self.supported_countries}
        
        print(f"CountryMapper đã được thiết lập với {len(self.supported_countries)} quốc gia.")
    
    def setup_aliases(self):
        """Thiết lập các tên gọi khác cho quốc gia. Có thể mở rộng thêm."""
        # Đây là một ví dụ, bạn có thể mở rộng danh sách này
        self.country_aliases = {
            "united states": "United States", "usa": "United States", "us": "United States", "america": "United States", "mỹ": "United States", "my": "United States", "hoa kỳ": "United States", "hoa ky": "United States",
            "china": "China", "trung quốc": "China", "trung quoc": "China", "trung": "China",
            "india": "India", "ấn độ": "India", "an do": "India",
            "brazil": "Brazil", "brasil": "Brazil", "ba tây": "Brazil", "ba tay": "Brazil",
            "russia": "Russia", "nga": "Russia", "liên bang nga": "Russia", "lien bang nga": "Russia",
            "japan": "Japan", "nhật bản": "Japan", "nhat ban": "Japan", "nhật": "Japan", "nhat": "Japan",
            "germany": "Germany", "đức": "Germany", "duc": "Germany",
            "united kingdom": "United Kingdom", "uk": "United Kingdom", "britain": "United Kingdom", "anh": "United Kingdom", "vương quốc anh": "United Kingdom", "vuong quoc anh": "United Kingdom",
            "france": "France", "pháp": "France", "phap": "France",
            "italy": "Italy", "ý": "Italy", "y": "Italy", "italia": "Italy",
            "south korea": "South Korea", "korea": "South Korea", "hàn quốc": "South Korea", "han quoc": "South Korea",
            "canada": "Canada", "ca-na-đa": "Canada", "ca-na-da": "Canada",
            "australia": "Australia", "úc": "Australia", "uc": "Australia", "châu úc": "Australia", "chau uc": "Australia",
            "mexico": "Mexico", "mê-hi-cô": "Mexico", "me-hi-co": "Mexico",
            "indonesia": "Indonesia", "in-đô-nê-xi-a": "Indonesia", "in-do-ne-xi-a": "Indonesia",
            "turkey": "Turkey", "thổ nhĩ kỳ": "Turkey", "tho nhi ky": "Turkey",
            "iran": "Iran", "i-ran": "Iran",
            "saudi arabia": "Saudi Arabia", "ả rập saudi": "Saudi Arabia", "a rap saudi": "Saudi Arabia",
            "thailand": "Thailand", "thái lan": "Thailand", "thai lan": "Thailand",
            "vietnam": "Vietnam", "việt nam": "Vietnam", "viet nam": "Vietnam",
            "malaysia": "Malaysia", "ma-lai-xi-a": "Malaysia", "mã lai": "Malaysia", "ma lai": "Malaysia",
            "singapore": "Singapore", "sin-ga-po": "Singapore", "xin-ga-po": "Singapore",
            "philippines": "Philippines", "phi-líp-pin": "Philippines", "phi-lip-pin": "Philippines",
            "egypt": "Egypt", "ai cập": "Egypt", "ai cap": "Egypt",
            "south africa": "South Africa", "nam phi": "South Africa", "nam africa": "South Africa",
            "nigeria": "Nigeria", "ni-giê-ri-a": "Nigeria", "ni-gie-ri-a": "Nigeria"
        }
        # Thêm các alias tự động từ tên quốc gia nếu cần
        for country in self.supported_countries:
            self.country_aliases[country.lower()] = country

    def find_country(self, text):
        """Tìm quốc gia từ text input"""
        text_lower = text.lower().strip()
        
        # Ưu tiên tìm kiếm khớp chính xác với tên quốc gia được hỗ trợ
        for country in self.supported_countries:
            if country.lower() in text_lower:
                # Kiểm tra nếu tên quốc gia là một từ riêng biệt hoặc được bao quanh bởi khoảng trắng/dấu câu
                pattern = r'\b' + re.escape(country.lower()) + r'\b'
                if re.search(pattern, text_lower):
                    return country
        
        # Sau đó kiểm tra aliases
        for alias, country in self.country_aliases.items():
            if alias in text_lower:
                # Kiểm tra nếu alias là một từ riêng biệt hoặc được bao quanh bởi khoảng trắng/dấu câu
                pattern = r'\b' + re.escape(alias) + r'\b'
                if re.search(pattern, text_lower):
                    return country
        
        return None
    
    def get_country_id(self, country):
        """Lấy ID của quốc gia cho embedding"""
        return self.country_mapping.get(country, None)
    
    def get_supported_countries(self):
        """Lấy danh sách quốc gia được hỗ trợ"""
        return self.supported_countries.copy()
    
    def get_country_info(self):
        """Lấy thông tin chi tiết về các quốc gia được hỗ trợ"""
        if self.data is None:
            return {}
        
        info = {}
        for country in self.supported_countries:
            country_data = self.data[self.data["location"] == country]
            if not country_data.empty:
                latest = country_data.iloc[-1]
                info[country] = {
                    "total_cases": latest.get("total_cases", 0),
                    "total_deaths": latest.get("total_deaths", 0),
                    "continent": latest.get("continent", "Unknown"),
                    "data_points": len(country_data)
                }
        return info
    
    def suggest_alternatives(self, text):
        """Gợi ý quốc gia thay thế khi không tìm thấy"""
        text_lower = text.lower()
        suggestions = []
        
        # Tìm các quốc gia có tên tương tự
        for country in self.supported_countries:
            if any(word in country.lower() for word in text_lower.split()):
                suggestions.append(country)
        
        # Nếu không có gợi ý, trả về tất cả
        if not suggestions:
            suggestions = self.supported_countries[:5]  # 5 quốc gia đầu tiên
        
        return suggestions




