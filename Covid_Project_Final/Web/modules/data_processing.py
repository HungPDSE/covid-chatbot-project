import pandas as pd
import streamlit as st
from pathlib import Path

@st.cache_data(ttl=3600)
def load_data():
    """Tải và xử lý dữ liệu COVID-19"""
    try:
        data_path = Path(__file__).parent.parent / "data" / "covid_cleaned_country_data.csv"
        df = pd.read_csv(data_path)
        df["date"] = pd.to_datetime(df["date"])
        
        # Tính toán các metrics bổ sung
        df["case_fatality_rate"] = (df["total_deaths"] / df["total_cases"] * 100).fillna(0)
        df["vaccination_rate"] = df["people_fully_vaccinated_per_hundred"].fillna(0)
        df["cases_per_million"] = df["total_cases_per_million"].fillna(0)
        df["new_cases_per_million"] = df["new_cases_per_million"].fillna(0)
        
        df.replace([float('inf'), float('-inf')], 0, inplace=True)
        
        return df
    except FileNotFoundError:
        st.error(f"Không tìm thấy tệp dữ liệu tại: {data_path}. Vui lòng đảm bảo file có tên đúng và nằm trong thư mục data/.")
        return None
