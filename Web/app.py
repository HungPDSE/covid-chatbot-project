# app.py
import streamlit as st
import pandas as pd
from datetime import timedelta
import os

# Import các module cần thiết
from modules.data_processing import load_data
from modules.utils import create_animated_metric_card
from modules.visualization import show_enhanced_time_trends, show_enhanced_world_map, show_enhanced_comparative_analysis
from modules.overview_analysis import show_overview_analysis
from modules.chatbot import show_chatbot_ui

#Tối ưu: Cache dữ liệu để tăng tốc độ
@st.cache_data
def get_data():
    df = load_data()
    return df

def main():
    st.set_page_config(
        page_title="COVID-19 Global Dashboard",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Tải CSS
    try:
        css_path = os.path.join(os.path.dirname(__file__), "styles", "custom.css")
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning("Không tìm thấy file styles/custom.css.")
        
    st.markdown("<h1 class=\"main-header\">COVID-19 Global Dashboard</h1>", unsafe_allow_html=True)
    
    df = get_data()
    if df is None:
        st.error("Lỗi nghiêm trọng: Không thể tải dữ liệu. Vui lòng kiểm tra file data.")
        return
    
    #Sidebar và bộ lọc (Đã cập nhật logic)
    with st.sidebar:
        st.markdown("<div class=\"sidebar-content\">", unsafe_allow_html=True)
        st.title("Bộ điều khiển")
        
        continents = ["Tất cả"] + sorted(df["continent"].dropna().unique().tolist())
        selected_continent = st.selectbox("1. Chọn châu lục:", continents)
        
        if selected_continent != "Tất cả":
            available_countries = sorted(df[df["continent"] == selected_continent]["location"].unique().tolist())
            countries_options = ["Tất cả quốc gia"] + available_countries
        else:
            countries_options = ["Toàn thế giới"] + sorted(df["location"].unique().tolist())

        selected_location = st.selectbox("2. Chọn quốc gia:", countries_options)
        
        st.subheader("Khoảng thời gian")
        time_preset = st.radio("Chọn nhanh:", ["Tùy chỉnh", "30 ngày qua", "90 ngày qua", "1 năm qua", "Toàn bộ"])
        
        min_date, max_date = df["date"].min(), df["date"].max()
        
        if time_preset == "30 ngày qua":
            start_date, end_date = max_date - timedelta(days=29), max_date
        elif time_preset == "90 ngày qua":
            start_date, end_date = max_date - timedelta(days=89), max_date
        elif time_preset == "1 năm qua":
            start_date, end_date = max_date - timedelta(days=364), max_date
        elif time_preset == "Toàn bộ":
            start_date, end_date = min_date, max_date
        else:
            date_range = st.date_input("Chọn khoảng thời gian:", value=(min_date, max_date), min_value=min_date, max_value=max_date)
            if len(date_range) == 2:
                start_date, end_date = date_range
            else:
                start_date, end_date = min_date, max_date
                
        st.info(f"Dữ liệu từ {start_date.strftime('%d/%m/%Y')} đến {end_date.strftime('%d/%m/%Y')}")
        st.markdown("</div>", unsafe_allow_html=True)

    #Lọc dữ liệu chính
    filtered_df = df.copy()
    if selected_continent != "Tất cả":
        filtered_df = filtered_df[filtered_df["continent"] == selected_continent]
    if selected_location not in ["Toàn thế giới", "Tất cả quốc gia"]:
        filtered_df = filtered_df[filtered_df["location"] == selected_location]
    filtered_df = filtered_df[(filtered_df["date"] >= pd.to_datetime(start_date)) & (filtered_df["date"] <= pd.to_datetime(end_date))]

    if filtered_df.empty:
        st.warning("Không có dữ liệu cho lựa chọn của bạn.")
        return

    #KPI Dashboard
    st.markdown("## Bảng điều khiển KPI")
    
    max_data_per_country = filtered_df.loc[filtered_df.groupby('location')['total_cases'].idxmax()]
    
    total_cases = max_data_per_country["total_cases"].sum()
    total_deaths = max_data_per_country["total_deaths"].sum()
    total_vaccinations = max_data_per_country["total_vaccinations"].sum()
    countries_affected = filtered_df["location"].nunique()
    
    
    # Cách tính có trọng số theo dân số vẫn được giữ lại để đảm bảo độ chính xác
    if not max_data_per_country.empty and max_data_per_country['population'].sum() > 0:
        avg_vaccination_rate = (max_data_per_country['people_fully_vaccinated'].sum() / max_data_per_country['population'].sum() * 100)
    else:
        avg_vaccination_rate = 0
        
    mortality_rate = (total_deaths / total_cases * 100) if total_cases > 0 else 0

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown(create_animated_metric_card("Tổng ca nhiễm", total_cases), unsafe_allow_html=True)
    with col2:
        st.markdown(create_animated_metric_card("Tổng ca tử vong", total_deaths), unsafe_allow_html=True)
    with col3:
        st.markdown(create_animated_metric_card("Tổng lượt tiêm", total_vaccinations), unsafe_allow_html=True)
    with col4:
        st.markdown(create_animated_metric_card("Quốc gia", countries_affected), unsafe_allow_html=True)
    with col5:
        st.markdown(create_animated_metric_card("Tỷ lệ tiêm chủng TB (%)", avg_vaccination_rate), unsafe_allow_html=True)
        
    st.markdown(f"""<div class="insight-box"><h4>Thông tin chi tiết</h4><p>Tỷ lệ tử vong (CFR): <strong>{mortality_rate:.2f}%</strong> | Tỷ lệ tiêm chủng trung bình (có trọng số): <strong>{avg_vaccination_rate:.1f}%</strong> | Quốc gia được phân tích: <strong>{countries_affected}</strong></p></div>""", unsafe_allow_html=True)

    #Tabs 
    tabs = st.tabs([
        " Xu hướng theo thời gian", 
        " Bản đồ thế giới", 
        " Phân tích tổng quan",
        " So sánh quốc gia",
        "🤖 Chatbot AI"
    ])
    
    with tabs[0]:
        show_enhanced_time_trends(filtered_df)
    with tabs[1]:
        show_enhanced_world_map(df) 
    with tabs[2]:
        show_overview_analysis(df)
    with tabs[3]:
        show_enhanced_comparative_analysis(df)
    with tabs[4]:
        show_chatbot_ui()

if __name__ == "__main__":
    main()
