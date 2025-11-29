# modules/overview_analysis.py
import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from pywaffle import Waffle

def create_waffle_chart(data_dict, title):
    fig = plt.figure(
        FigureClass=Waffle,
        rows=5,
        values=data_dict,
        colors=("#232066", "#983D3D", "#DCB732"),
        title={'label': title, 'loc': 'left', 'fontsize': 16},
        labels=[f"{k} ({v}%)" for k, v in data_dict.items()],
        legend={'loc': 'lower left', 'bbox_to_anchor': (0, -0.4), 'ncol': len(data_dict), 'framealpha': 0},
        figsize=(10, 5)
    )
    fig.gca().set_facecolor('#EEEEEE')
    fig.set_facecolor('#EEEEEE')
    return fig

def show_overview_analysis(df):
    st.markdown("###  Phân tích tổng quan theo khu vực")
    st.info("Lưu ý: Tất cả các phân tích trong tab này đều dựa trên **số liệu cao nhất (max)** từng được ghi nhận của mỗi quốc gia để đảm bảo tính chính xác.")

    if df.empty:
        st.warning("Không có dữ liệu để thực hiện phân tích.")
        return

    # DỌN DẸP: Sử dụng trực tiếp tên cột có sẵn
    metric_options = {
        "Tổng ca nhiễm": "total_cases",
        "Tổng ca tử vong": "total_deaths",
        "Tỷ lệ tiêm chủng đầy đủ (%)": "people_fully_vaccinated_per_hundred", # <-- THAY ĐỔI Ở ĐÂY
        "Ca nhiễm/triệu dân": "total_cases_per_million",
        "Tử vong/triệu dân": "total_deaths_per_million"
    }
    selected_metric_label = st.selectbox(
        "Chọn chỉ số để phân tích tổng quan:", 
        list(metric_options.keys()),
        key="overview_metric_select"
    )
    selected_metric = metric_options[selected_metric_label]

    # Lấy hàng có giá trị MAX của chỉ số được chọn cho mỗi quốc gia
    max_metric_data = df.loc[df.groupby('location')[selected_metric].idxmax()]

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"####  Top 10 quốc gia có {selected_metric_label} cao nhất")
        top_10_countries = max_metric_data.nlargest(10, selected_metric)
        fig_bar = px.bar(
            top_10_countries, x="location", y=selected_metric,
            color=selected_metric, color_continuous_scale='Viridis',
            title=f"Top 10 quốc gia - {selected_metric_label}",
            labels={"location": "Quốc gia"}
        )
        fig_bar.update_layout(xaxis_title="", yaxis_title=selected_metric_label)
        st.plotly_chart(fig_bar, use_container_width=True)

    with col2:
        st.markdown(f"####  Phân bổ {selected_metric_label} theo châu lục")
        continent_data = max_metric_data.groupby("continent")[selected_metric].sum().reset_index()
        fig_pie = px.pie(
            continent_data, names="continent", values=selected_metric,
            title=f"Tỷ trọng {selected_metric_label} theo châu lục",
            hole=0.3
        )
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown(f"####  Phân bổ của '{selected_metric_label}' tại các châu lục")
    fig_box = px.box(
        max_metric_data.dropna(subset=[selected_metric, 'continent']),
        x="continent", y=selected_metric, color="continent",
        title=f"Phân tích phân tán của {selected_metric_label}",
        labels={"continent": "Châu lục", selected_metric: selected_metric_label}
    )
    st.plotly_chart(fig_box, use_container_width=True)

    # Biểu đồ WAFFLE
    st.markdown("---")
    st.markdown("###  Phân tích tỷ lệ tiêm chủng toàn cầu (Waffle Chart)")

    max_vax_data = df.loc[df.groupby('location')['people_vaccinated'].idxmax()]
    
    global_population = max_vax_data['population'].sum()
    global_fully_vaccinated = max_vax_data['people_fully_vaccinated'].sum()
    global_partially_vaccinated = max_vax_data['people_vaccinated'].sum() - global_fully_vaccinated

    fully_vaccinated_pct = (global_fully_vaccinated / global_population) * 100 if global_population > 0 else 0
    partially_vaccinated_pct = (global_partially_vaccinated / global_population) * 100 if global_population > 0 else 0
    unvaccinated_pct = 100 - fully_vaccinated_pct - partially_vaccinated_pct

    partially_vaccinated_pct = max(0, partially_vaccinated_pct)
    unvaccinated_pct = max(0, unvaccinated_pct)

    data_for_waffle = {
        'Đã tiêm đủ': round(fully_vaccinated_pct),
        'Tiêm 1 mũi': round(partially_vaccinated_pct),
        'Chưa tiêm': round(unvaccinated_pct)
    }
    
    diff = 100 - sum(data_for_waffle.values())
    if diff != 0:
        max_key = max(data_for_waffle, key=data_for_waffle.get)
        data_for_waffle[max_key] += diff

    st.write(f"Dữ liệu được tính toán dựa trên **số liệu tiêm chủng cao nhất** của mỗi quốc gia. Mỗi ô vuông tương ứng với 1% dân số.")
    
    fig_waffle = create_waffle_chart(
        data_for_waffle,
        'Tỷ lệ tiêm chủng COVID-19 trên toàn cầu'
    )
    st.pyplot(fig_waffle)
