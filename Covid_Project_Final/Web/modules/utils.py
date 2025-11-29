import streamlit as st

def create_animated_metric_card(title, value, delta=None, delta_color="normal"):
    """Tạo thẻ metric với hiệu ứng động"""
    if isinstance(value, float):
        value_str = f"{value:,.2f}"
    else:
        value_str = f"{value:,.0f}"

    delta_html = ""
    if delta:
        color = "#28a745" if delta_color == "normal" else "#dc3545"
        delta_html = f'<div style="font-size: 0.8rem; color: {color}; margin-top: 0.5rem;">{"↗" if delta > 0 else "↘"} {abs(delta):,.0f}</div>'
    
    return f"""
    <div class="metric-card">
        <div class="metric-title">{title}</div>
        <div class="metric-value">{value_str}</div>
        {delta_html}
    </div>
    """
