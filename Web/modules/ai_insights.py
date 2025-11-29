import streamlit as st
import pandas as pd

def show_ai_insights(df):
    """Hiển thị insights và phân tích AI"""
    st.markdown("### 🎯 AI Insights & Dự báo")
    
    # Tính toán các insights tự động
    latest_data = df.groupby("location").last().reset_index()
    
    # Top insights
    insights = []
    
    # Insight 1: Quốc gia có tỷ lệ tử vong cao nhất
    highest_mortality = latest_data.loc[latest_data["case_fatality_rate"].idxmax()]
    insights.append(f"🔴 Tỷ lệ tử vong cao nhất: {highest_mortality['location']} ({highest_mortality['case_fatality_rate']:.2f}%)")
    
    # Insight 2: Quốc gia có tỷ lệ tiêm chủng cao nhất
    highest_vaccination = latest_data.loc[latest_data["vaccination_rate"].idxmax()]
    insights.append(f"💉 Tiêm chủng tốt nhất: {highest_vaccination['location']} ({highest_vaccination['vaccination_rate']:.1f}%)")
    
    # Insight 3: Tương quan GDP và ca nhiễm
    correlation = latest_data["gdp_per_capita"].corr(latest_data["cases_per_million"])
    insights.append(f"💰 Tương quan GDP-Ca nhiễm: {correlation:.3f} ({'Dương' if correlation > 0 else 'Âm'})")
    
    # Hiển thị insights
    for insight in insights:
        st.markdown(f"""
        <div class="insight-box">
            {insight}
        </div>
        """, unsafe_allow_html=True)
    
    # Dự báo đơn giản
    st.markdown("#### 📈 Dự báo xu hướng")
    
    # Tính toán xu hướng cho 7 ngày tới
    daily_global = df.groupby("date").agg({
        "new_cases": "sum",
        "new_deaths": "sum"
    }).reset_index().tail(30)  # Lấy 30 ngày gần nhất
    
    if len(daily_global) > 7:
        # Tính toán moving average
        daily_global["cases_ma7"] = daily_global["new_cases"].rolling(7).mean()
        daily_global["deaths_ma7"] = daily_global["new_deaths"].rolling(7).mean()
        
        # Tính toán trend
        recent_cases_trend = daily_global["cases_ma7"].tail(7).mean() - daily_global["cases_ma7"].tail(14).head(7).mean()
        recent_deaths_trend = daily_global["deaths_ma7"].tail(7).mean() - daily_global["deaths_ma7"].tail(14).head(7).mean()
        
        col1, col2 = st.columns(2)
        
        with col1:
            trend_color = "🟢" if recent_cases_trend < 0 else "🔴"
            st.metric(
                "Xu hướng ca nhiễm (7 ngày)",
                f"{recent_cases_trend:+,.0f}",
                delta=f"{recent_cases_trend:+,.0f}"
            )
        
        with col2:
            trend_color = "🟢" if recent_deaths_trend < 0 else "🔴"
            st.metric(
                "Xu hướng ca tử vong (7 ngày)",
                f"{recent_deaths_trend:+,.0f}",
                delta=f"{recent_deaths_trend:+,.0f}"
            )
    
    # Recommendations
    st.markdown("#### 💡 Khuyến nghị dựa trên dữ liệu")
    
    recommendations = [
        "🎯 Tăng cường tiêm chủng tại các quốc gia có tỷ lệ thấp",
        "🏥 Cải thiện hệ thống y tế tại các quốc gia có tỷ lệ tử vong cao",
        "📊 Theo dõi chặt chẽ các quốc gia có xu hướng tăng ca nhiễm",
        "🌍 Hợp tác quốc tế trong việc chia sẻ vaccine và kinh nghiệm"
    ]
    
    for rec in recommendations:
        st.markdown(f"- {rec}")
