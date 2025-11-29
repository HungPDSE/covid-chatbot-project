import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def show_enhanced_advanced_analysis(df):
    """Hiển thị phân tích nâng cao với ML insights"""
    st.markdown("### 🔬 Phân tích nâng cao & Machine Learning")
    
    if df.empty:
        st.warning("Không có dữ liệu để thực hiện phân tích nâng cao.")
        return

    latest_data = df.groupby("location").last().reset_index()
    
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### GDP, Tuổi thọ và Ca nhiễm")
        fig_3d = px.scatter_3d(
            latest_data.dropna(subset=["gdp_per_capita", "life_expectancy", "total_cases", "population", "continent"]),
            x="gdp_per_capita",
            y="life_expectancy",
            z="total_cases",
            size="population",
            color="continent",
            hover_name="location",
            title="Phân tích 3D: GDP vs Tuổi thọ vs Ca nhiễm",
            labels={"gdp_per_capita": "GDP/người", "life_expectancy": "Tuổi thọ", "total_cases": "Tổng ca nhiễm"}
        )
        fig_3d.update_layout(height=500)
        st.plotly_chart(fig_3d, use_container_width=True)
    
    with col2:
        st.markdown("#### Hiệu quả tiêm chủng")
        vaccination_data = latest_data[
            (latest_data["people_fully_vaccinated_per_hundred"] > 0) &
            (latest_data["total_cases"] > 1000)
        ].copy()
        
        if not vaccination_data.empty:
            fig_vax = px.scatter(
                vaccination_data,
                x="people_fully_vaccinated_per_hundred",
                y="new_cases_per_million",
                size="population",
                color="continent",
                hover_name="location",
                title="Tương quan: Tỷ lệ tiêm chủng và Ca nhiễm mới",
                trendline="ols",
                labels={"people_fully_vaccinated_per_hundred": "Tỷ lệ tiêm chủng (%)", "new_cases_per_million": "Ca mới/triệu dân"}
            )
            fig_vax.update_layout(height=500)
            st.plotly_chart(fig_vax, use_container_width=True)

    st.markdown("#### 🎯 Phân nhóm quốc gia theo đặc điểm COVID-19")
    cluster_features = ["cases_per_million", "vaccination_rate", "case_fatality_rate", "gdp_per_capita"]
    cluster_data = latest_data[cluster_features + ["location", "continent"]].dropna()
    
    if len(cluster_data) > 10:
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(cluster_data[cluster_features])
        
        kmeans = KMeans(n_clusters=4, random_state=42, n_init='auto')
        cluster_data["cluster"] = kmeans.fit_predict(scaled_features)
        
        fig_cluster = px.scatter(
            cluster_data,
            x="cases_per_million",
            y="vaccination_rate",
            color="cluster",
            size="gdp_per_capita",
            hover_name="location",
            title="Phân nhóm quốc gia (K-means)",
            labels={"cases_per_million": "Ca nhiễm/triệu dân", "vaccination_rate": "Tỷ lệ tiêm chủng (%)"}
        )
        st.plotly_chart(fig_cluster, use_container_width=True)
        
        st.markdown("#### 📋 Đặc điểm các nhóm quốc gia")
        cluster_summary = cluster_data.groupby("cluster").agg({
            "cases_per_million": "mean", "vaccination_rate": "mean",
            "case_fatality_rate": "mean", "gdp_per_capita": "mean",
            "location": "count"
        }).round(2)
        cluster_summary.columns = ["Ca/triệu dân (TB)", "Tỷ lệ tiêm chủng (TB)", "Tỷ lệ tử vong (TB)", "GDP/người (TB)", "Số quốc gia"]
        st.dataframe(cluster_summary, use_container_width=True)

def show_ai_insights(df):
    """Hiển thị insights và phân tích AI"""
    st.markdown("### 🎯 Insights từ Dữ liệu")
    
    if df.empty:
        st.warning("Không có dữ liệu để tạo insights.")
        return

    latest_data = df.groupby("location").last().reset_index()
    
    insights = []
    
    highest_mortality = latest_data.loc[latest_data["case_fatality_rate"].idxmax()]
    insights.append(f"🔴 **Tỷ lệ tử vong cao nhất:** {highest_mortality['location']} ({highest_mortality['case_fatality_rate']:.2f}%)")
    
    highest_vaccination = latest_data.loc[latest_data["vaccination_rate"].idxmax()]
    insights.append(f"💉 **Tiêm chủng tốt nhất:** {highest_vaccination['location']} ({highest_vaccination['vaccination_rate']:.1f}%)")
    
    correlation = latest_data["gdp_per_capita"].corr(latest_data["cases_per_million"])
    insights.append(f"💰 **Tương quan GDP-Ca nhiễm:** {correlation:.3f} (Cho thấy mối liên hệ {'yếu' if abs(correlation) < 0.3 else 'trung bình' if abs(correlation) < 0.7 else 'mạnh'} giữa GDP và số ca nhiễm trên triệu dân)")
    
    for insight in insights:
        st.markdown(f"<div class='insight-box'>{insight}</div>", unsafe_allow_html=True)
    
    st.markdown("#### 💡 Khuyến nghị dựa trên dữ liệu")
    recommendations = [
        "Tập trung nguồn lực hỗ trợ các quốc gia có tỷ lệ tử vong cao nhưng GDP thấp.",
        "Thúc đẩy chiến dịch tiêm chủng ở các khu vực có tỷ lệ bao phủ còn thấp.",
        "Theo dõi chặt chẽ các biến thể mới tại các quốc gia có số ca nhiễm tăng đột biến.",
        "Phân tích sâu hơn mối quan hệ giữa các yếu tố kinh tế-xã hội và hiệu quả phòng chống dịch."
    ]
    for rec in recommendations:
        st.markdown(f"- {rec}")
