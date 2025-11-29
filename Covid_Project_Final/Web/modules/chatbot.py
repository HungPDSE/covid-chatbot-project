# modules/chatbot.py
import streamlit as st
from streamlit_chat import message
from google.oauth2 import service_account
from google.cloud import dialogflow_v2 as dialogflow
import uuid
import json
import os
import re
from datetime import datetime, timedelta
from .prediction_service import get_prediction_service
from .country_mapper import CountryMapper 
from .data_query_service import get_data_query_service

# --- Cấu hình (Phiên bản đơn giản) ---
KEY_PATH = "dialogflow_key.json"

try:
    with open(KEY_PATH, "r") as f:
        credentials_info = json.load(f)
        project_id = credentials_info.get("project_id")

    creds = service_account.Credentials.from_service_account_file(KEY_PATH)
    session_client = dialogflow.SessionsClient(credentials=creds)

except FileNotFoundError:
    st.error(f"Lỗi: Không tìm thấy file \'{KEY_PATH}\'. Vui lòng đặt file key vào thư mục gốc của dự án. Chatbot sẽ không hoạt động.")
    session_client = None
    project_id = None
except Exception as e:
    st.error(f"Lỗi khi khởi tạo chatbot: {e}")
    session_client = None
    project_id = None

if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

def detect_intent_texts(text, language_code='vi'):
    """Gửi một truy vấn văn bản đến Dialogflow và trả về phản hồi."""
    
    # Kiểm tra xem có phải là yêu cầu truy vấn dữ liệu không (Ưu tiên xử lý trước)
    data_query_response = handle_data_query_request(text)
    if data_query_response:
        return data_query_response

    # Kiểm tra xem có phải là yêu cầu dự đoán không
    prediction_response = handle_prediction_request(text)
    if prediction_response:
        return prediction_response
    
    if not session_client or not project_id:
        return "Xin lỗi, chatbot đang gặp sự cố kỹ thuật."

    session = session_client.session_path(project_id, st.session_state.session_id)
    text_input = dialogflow.TextInput(text=text, language_code=language_code)
    query_input = dialogflow.QueryInput(text=text_input)

    try:
        response = session_client.detect_intent(
            request={"session": session, "query_input": query_input}
        )
        return response.query_result.fulfillment_text
    except Exception as e:
        print(f"Lỗi khi gọi Dialogflow API: {e}")
        return "Oops! Có lỗi xảy ra khi kết nối tới chatbot."

def handle_data_query_request(text):
    """Xử lý yêu cầu truy vấn dữ liệu từ người dùng"""
    text_lower = text.lower()
    
    # Các từ khóa để nhận diện yêu cầu truy vấn dữ liệu
    data_query_keywords = [
        "dữ liệu", "du lieu", "data","tổng quan", 
        "tong quan", "overview","số liệu", "so lieu", 
        "statistics", "thống kê", "thong ke"
    ]
    
    # Kiểm tra xem có từ khóa truy vấn dữ liệu không
    has_data_query_keyword = any(keyword in text_lower for keyword in data_query_keywords)
    
    # Loại trừ các từ khóa dự đoán để tránh nhầm lẫn
    prediction_keywords_for_exclusion = [
        "dự đoán", "du doan", "predict", "forecast", 
        "dự báo", "du bao", "tương lai", "tuong lai",
        "ngày tới", "ngay toi", "sắp tới", "sap toi"
    ]
    is_prediction_query = any(keyword in text_lower for keyword in prediction_keywords_for_exclusion)

    # Nếu có từ khóa truy vấn dữ liệu nhưng cũng có từ khóa dự đoán, thì đây có thể là yêu cầu dự đoán
    if has_data_query_keyword and is_prediction_query:
        return None
    
    # Nếu không có từ khóa truy vấn dữ liệu, hoặc là yêu cầu dự đoán, thì không xử lý ở đây
    if not has_data_query_keyword:
        return None
    
    try:
        # Lấy data query service
        data_service = get_data_query_service()
        
        # Kiểm tra xem có yêu cầu tổng quan không
        overview_keywords = ["tổng quan", "tong quan", "overview", "tình hình chung", "tinh hinh chung"]
        if any(keyword in text_lower for keyword in overview_keywords):
            return data_service.get_overview_data()
        
        pred_service = get_prediction_service()
        
        # Trích xuất tên quốc gia từ text
        country = pred_service.country_mapper.find_country(text)
        
        # Trích xuất ngày cụ thể (nếu có)
        target_date = extract_date_from_text(text)
        
        if country and target_date:
            # Truy vấn dữ liệu cho quốc gia và ngày cụ thể
            return data_service.get_data_by_date_and_country(country, target_date)
        elif country:
            # Truy vấn dữ liệu mới nhất cho quốc gia
            return data_service.get_country_data_summary(country)
        else:
            # Nếu không có quốc gia cụ thể, trả về tổng quan
            return data_service.get_overview_data()
        
    except Exception as e:
        return f"Lỗi khi truy vấn dữ liệu: {e}"

def extract_date_from_text(text):
    """Trích xuất ngày cụ thể từ text input (nếu có)"""
    text_lower = text.lower()
    
    # Pattern cho định dạng ngày dd/mm/yyyy hoặc dd/mm/yy
    date_patterns = [
        r'(\d{1,2})[/-](\d{1,2})[/-](\d{4})',  # dd/mm/yyyy
        r'(\d{1,2})[/-](\d{1,2})[/-](\d{2})'   # dd/mm/yy
    ]
    
    for pattern in date_patterns:
        match = re.search(pattern, text_lower)
        if match:
            day, month, year = match.groups()
            # Xử lý năm 2 chữ số
            if len(year) == 2:
                year = "20" + year if int(year) < 50 else "19" + year # Giả định năm 20xx cho <50, 19xx cho >=50
            try:
                return datetime(int(year), int(month), int(day)).date()
            except ValueError:
                continue
                
    # Pattern cho các từ khóa như "ngày mai", "hôm sau", "tuần tới", "tháng tới"
    if "ngày mai" in text_lower or "ngay mai" in text_lower:
        return (datetime.now() + timedelta(days=1)).date()
    if "ngày kia" in text_lower or "ngay kia" in text_lower:
        return (datetime.now() + timedelta(days=2)).date()
    if "tuần tới" in text_lower or "tuan toi" in text_lower:
        return (datetime.now() + timedelta(weeks=1)).date()
    if "tháng tới" in text_lower or "thang toi" in text_lower:
        return (datetime.now() + timedelta(days=30)).date() # Ước lượng 30 ngày cho tháng tới

    return None

def handle_prediction_request(text):
    """Xử lý yêu cầu dự đoán từ người dùng"""
    text_lower = text.lower()
    
    # Các từ khóa để nhận diện yêu cầu dự đoán
    prediction_keywords = [
        "dự đoán", "du doan", "predict", "forecast", 
        "dự báo", "du bao", "tương lai", "tuong lai",
        "ngày tới", "ngay toi", "sắp tới", "sap toi",
        "ở ngày", "o ngay", "vào ngày", "vao ngay"
    ]
    
    # Kiểm tra xem có từ khóa dự đoán không
    has_prediction_keyword = any(keyword in text_lower for keyword in prediction_keywords)
    
    if not has_prediction_keyword:
        return None
    
    try:
        # Lấy prediction service
        pred_service = get_prediction_service()
        
        # Trích xuất tên quốc gia từ text sử dụng CountryMapper
        country = pred_service.country_mapper.find_country(text)
        if not country:
            available_countries = pred_service.country_mapper.get_supported_countries()
            return f"Tôi cần biết quốc gia để dự đoán. Các quốc gia được hỗ trợ: {', '.join(available_countries[:5])}... \n\nVí dụ: 'Dự đoán COVID cho {available_countries[0] if available_countries else 'Vietnam'} 5 ngày tới'"
        
        # Trích xuất ngày cụ thể (nếu có)
        target_date = extract_date_from_text(text)
        
        days_ahead = 3 # Mặc định là 3 ngày
        if target_date:
            # Tính số ngày từ hôm nay đến ngày mục tiêu
            today = datetime.now().date()
            delta = target_date - today
            days_ahead = max(1, delta.days) # Đảm bảo ít nhất 1 ngày
        else:
            # Nếu không có ngày cụ thể, trích xuất số ngày tới
            days_ahead = extract_days_from_text(text)
        
        # Thực hiện dự đoán
        predictions, error = pred_service.predict_cases(country, target_date, days_ahead)
        
        if error:
            return f"Lỗi dự đoán: {error}"
        
        # Lấy thông tin độ tin cậy
        confidence_level, confidence_msg = pred_service.get_prediction_confidence(country, target_date)
        
        # Định dạng phản hồi
        response = pred_service.format_prediction_response(
            country, predictions, confidence_level, confidence_msg
        )
        
        return response
        
    except Exception as e:
        return f"Lỗi khi xử lý yêu cầu dự đoán: {e}"

def extract_days_from_text(text):
    """Trích xuất số ngày dự đoán từ text"""
    # Tìm số ngày trong text
    
    # Pattern để tìm số + "ngày"
    patterns = [
        r'(\d+)\s*ngày',
        r'(\d+)\s*day',
        r'(\d+)\s*days'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text.lower())
        if match:
            days = int(match.group(1))
            # Giới hạn số ngày dự đoán
            return min(days, 30)  # Tối đa 30 ngày
    
    # Mặc định là 3 ngày nếu không tìm thấy
    return 3

def show_chatbot_ui():
    """Hiển thị giao diện người dùng cho chatbot."""
    st.markdown("Chatbot COVID-19 với Dự đoán AI")
    
    # Hướng dẫn sử dụng
    with st.expander("Hướng dẫn sử dụng", expanded=False):
        st.markdown("""
        **Chatbot này có thể:**
        - Trả lời các câu hỏi về dữ liệu COVID-19
        - **Dự đoán số ca bệnh** sử dụng AI model BiLSTM
        - **Truy vấn dữ liệu** từ tập dữ liệu COVID-19
        """)
    
    st.write("Bạn có thể hỏi tôi các câu hỏi về dữ liệu COVID-19 hoặc yêu cầu dự đoán!")

    if 'history' not in st.session_state:
        st.session_state['history'] = []
    
    if st.button("Xóa lịch sử chat"):
        st.session_state['history'] = []
        st.rerun()

    if st.session_state.history:
        for i, (user_msg, bot_msg) in enumerate(st.session_state.history):
            message(user_msg, is_user=True, key=f"user_{i}")
            message(bot_msg, key=f"bot_{i}")

    user_input = st.chat_input("Nhập câu hỏi của bạn...")

    if user_input:
        bot_response = detect_intent_texts(user_input)
        st.session_state.history.append((user_input, bot_response))
        st.rerun()



