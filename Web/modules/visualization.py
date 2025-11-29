import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# Hàm định dạng số lớn
def format_large_number(num):
    """Định dạng một số lớn thành chuỗi có đơn vị (Nghìn, Triệu, Tỷ)."""
    if pd.isna(num):
        return "N/A"
    if num >= 1_000_000_000:
        return f"{num / 1_000_000_000:,.2f} Tỷ"
    if num >= 1_000_000:
        return f"{num / 1_000_000:,.2f} Triệu"
    if num >= 1_000:
        return f"{num / 1_000:,.2f} Nghìn"
    return f"{num:,.0f}"

def plot_single_metric_trend(df, date_col, value_col, smoothed_col, title, y_axis_title, color_primary, color_secondary):
    """Hàm trợ giúp để vẽ một biểu đồ xu hướng cho một chỉ số duy nhất."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df[date_col], y=df[value_col], mode='lines', name='Hàng ngày',
        line=dict(color=color_secondary, width=1), fill='tonexty', opacity=0.3
    ))
    fig.add_trace(go.Scatter(
        x=df[date_col], y=df[smoothed_col], mode='lines', name='Trung bình 7 ngày',
        line=dict(color=color_primary, width=3)
    ))
    fig.update_layout(
        title=title, yaxis_title=y_axis_title, template="plotly_white", showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

def show_enhanced_time_trends(df):
    """Hiển thị các biểu đồ xu hướng được cải tiến, dễ nhìn hơn."""
    st.markdown("###  Phân tích xu hướng theo thời gian")
    
    if df.empty:
        st.warning("Không có dữ liệu để hiển thị xu hướng.")
        return

    daily_data = df.groupby("date").agg({
        "new_cases": "sum", "new_deaths": "sum", "new_vaccinations": "sum",
        "new_cases_smoothed": "sum", "new_deaths_smoothed": "sum", "new_vaccinations_smoothed": "sum"
    }).reset_index()

    #Biểu đồ 1: Ca nhiễm mới
    st.markdown("---")
    st.markdown("####  Phân tích ca nhiễm mới")
    col1, col2 = st.columns([3, 1])

    with col1:
        fig_cases = plot_single_metric_trend(
            daily_data, 'date', 'new_cases', 'new_cases_smoothed',
            'Xu hướng ca nhiễm mới hàng ngày', 'Số ca nhiễm', '#4ecdc4', '#a2dada'
        )
        st.plotly_chart(fig_cases, use_container_width=True)
    
    with col2:
        st.markdown("##### Thống kê chính")
        total_cases_period = daily_data['new_cases'].sum()
        max_cases_day = daily_data['new_cases'].max()
        avg_cases_day = daily_data['new_cases'].mean()
        # CẬP NHẬT: Áp dụng hàm định dạng
        st.metric("Tổng ca nhiễm trong kỳ", format_large_number(total_cases_period))
        st.metric("Cao nhất trong ngày", format_large_number(max_cases_day))
        st.metric("Trung bình mỗi ngày", format_large_number(avg_cases_day))

    #Biểu đồ 2: Ca tử vong mới
    st.markdown("---")
    st.markdown("####  Phân tích ca tử vong mới")
    col1, col2 = st.columns([3, 1])

    with col1:
        fig_deaths = plot_single_metric_trend(
            daily_data, 'date', 'new_deaths', 'new_deaths_smoothed',
            'Xu hướng ca tử vong mới hàng ngày', 'Số ca tử vong', '#ee5a52', '#f6b1ad'
        )
        st.plotly_chart(fig_deaths, use_container_width=True)

    with col2:
        st.markdown("##### Thống kê chính")
        total_deaths_period = daily_data['new_deaths'].sum()
        max_deaths_day = daily_data['new_deaths'].max()
        avg_deaths_day = daily_data['new_deaths'].mean()
        # CẬP NHẬT: Áp dụng hàm định dạng
        st.metric("Tổng ca tử vong trong kỳ", format_large_number(total_deaths_period))
        st.metric("Cao nhất trong ngày", format_large_number(max_deaths_day))
        st.metric("Trung bình mỗi ngày", format_large_number(avg_deaths_day))

    # --- Biểu đồ 3: Lượt tiêm chủng mới ---
    st.markdown("---")
    st.markdown("####  Phân tích tiêm chủng mới")
    col1, col2 = st.columns([3, 1])

    with col1:
        fig_vax = plot_single_metric_trend(
            daily_data, 'date', 'new_vaccinations', 'new_vaccinations_smoothed',
            'Xu hướng lượt tiêm chủng mới hàng ngày', 'Số lượt tiêm', '#26de81', '#a4f2c3'
        )
        st.plotly_chart(fig_vax, use_container_width=True)

    with col2:
        st.markdown("##### Thống kê chính")
        total_vax_period = daily_data['new_vaccinations'].sum()
        max_vax_day = daily_data['new_vaccinations'].max()
        avg_vax_day = daily_data['new_vaccinations'].mean()
        # CẬP NHẬT: Áp dụng hàm định dạng
        st.metric("Tổng lượt tiêm trong kỳ", format_large_number(total_vax_period))
        st.metric("Cao nhất trong ngày", format_large_number(max_vax_day))
        st.metric("Trung bình mỗi ngày", format_large_number(avg_vax_day))


def show_enhanced_world_map(df):
    """Hiển thị bản đồ thế giới tương tác với dữ liệu đã được lọc"""
    st.markdown("###  Bản đồ dịch tễ toàn cầu")
    
    if df.empty:
        st.warning("Không có dữ liệu để hiển thị bản đồ.")
        return

    col1, col2 = st.columns(2)
    with col1:
        metric_options = {
            "Tổng ca nhiễm": "total_cases",
            "Tổng ca tử vong": "total_deaths",
            "Tỷ lệ tiêm chủng (%)": "vaccination_rate",
            "Ca nhiễm/triệu dân": "cases_per_million"
        }
        selected_metric_label = st.selectbox("Chọn chỉ số:", list(metric_options.keys()))
        selected_metric = metric_options[selected_metric_label]
    
    with col2:
        color_scales = ['Plasma', 'Viridis', 'Cividis', 'Blues', 'Reds', 'Greens']
        selected_color_scale = st.selectbox("Chọn bảng màu:", color_scales)

    map_data = df.groupby("location").agg({
        selected_metric: "max",
        "iso_code": "first",
        "continent": "first"
    }).reset_index()

    fig = px.choropleth(
        map_data,
        locations="iso_code",
        color=selected_metric,
        hover_name="location",
        color_continuous_scale=selected_color_scale,
        title=f"Bản đồ thế giới: {selected_metric_label}"
    )
    fig.update_layout(height=600, geo=dict(bgcolor='rgba(0,0,0,0)'))
    st.plotly_chart(fig, use_container_width=True)


def show_enhanced_comparative_analysis(df):
    """Hiển thị phân tích so sánh giữa các quốc gia với các chỉ số đã được Việt hóa."""
    st.markdown("###  Phân tích so sánh đa quốc gia")
    st.info("So sánh diễn biến các chỉ số theo thời gian giữa các quốc gia bạn chọn.")
    
    countries = sorted(df["location"].unique().tolist())
    selected_countries = st.multiselect(
        "Chọn các quốc gia để so sánh:",
        countries,
        default=["Vietnam", "United States", "India", "Brazil"]
    )
    
    if not selected_countries:
        st.warning("Vui lòng chọn ít nhất một quốc gia.")
        return

    comp_df = df[df["location"].isin(selected_countries)]
    
    # THAY ĐỔI Ở ĐÂY: Việt hóa các lựa chọn
    metric_options_comp = {
        "Ca nhiễm mới/triệu dân (làm mịn)": "new_cases_smoothed_per_million",
        "Ca tử vong mới/triệu dân (làm mịn)": "new_deaths_smoothed_per_million",
        "Tỷ lệ tiêm chủng đầy đủ (%)": "people_fully_vaccinated_per_hundred",
        "Chỉ số nghiêm ngặt (Stringency Index)": "stringency_index"
    }
    
    # Hiển thị các key tiếng Việt cho người dùng lựa chọn
    selected_metric_label = st.selectbox(
        "Chọn chỉ số để so sánh:",
        list(metric_options_comp.keys()), # <-- Chỉ hiển thị các key đã Việt hóa
        key="compare_metric"
    )
    
    # Lấy tên cột tương ứng từ lựa chọn của người dùng
    metric_to_plot = metric_options_comp[selected_metric_label]

    fig = px.line(
        comp_df,
        x="date",
        y=metric_to_plot,
        color="location",
        # Cập nhật tiêu đề và nhãn của biểu đồ để hiển thị tiếng Việt
        title=f"So sánh '{selected_metric_label}' giữa các quốc gia",
        labels={
            metric_to_plot: selected_metric_label, # Nhãn trục Y
            'date': 'Ngày', 
            'location': 'Quốc gia'
        }
    )
    st.plotly_chart(fig, use_container_width=True)

