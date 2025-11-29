import numpy as np
import pandas as pd
from datetime import datetime, timedelta, date
import streamlit as st
from tensorflow.keras.models import load_model
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from .country_mapper import CountryMapper

class CovidPredictionService:
    def __init__(self):
        self.model = None
        self.data = None
        self.country_mapper = None
        self.scalers = {}
        self.load_model_and_data()

    def load_model_and_data(self):
        try:
            model_path = Path(__file__).parent.parent / "data" / "bilstm_covid19_model_with_emb.h5"
            if model_path.exists():
                self.model = load_model(str(model_path))
                st.success("Model BiLSTM đã được tải thành công!")
            else:
                st.error("Không tìm thấy file model BiLSTM")
                return

            data_path = Path(__file__).parent.parent / "data" / "Covid19_cleaned_to_model.csv"
            if data_path.exists():
                self.country_mapper = CountryMapper(data_path)
                self.data = self.country_mapper.data
                
                self._preprocess_dates()
                self._preprocess_and_fit_scalers()

                print(self.data.head())
                print(f"Khoảng thời gian dữ liệu: {self.data['date'].min()} đến {self.data['date'].max()}")
                
            else:
                st.error("Không tìm thấy file dữ liệu COVID")
        except Exception as e:
            st.error(f"Lỗi khi tải model/dữ liệu: {e}")
            print(f"Chi tiết lỗi: {e}")

    def _preprocess_dates(self):
        if self.data is None:
            return
            
        try:
            if 'date' in self.data.columns:
                date_formats = ['%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y', '%Y/%m/%d']
                
                for fmt in date_formats:
                    try:
                        self.data['date'] = pd.to_datetime(self.data['date'], format=fmt, errors='raise')
                        print(f"Thành công với format: {fmt}")
                        break
                    except (ValueError, TypeError):
                        continue
                else:
                    self.data['date'] = pd.to_datetime(self.data['date'], errors='coerce', dayfirst=True)
                
                initial_rows = len(self.data)
                self.data = self.data.dropna(subset=['date'])
                final_rows = len(self.data)
                
                if initial_rows != final_rows:
                    print(f"Đã loại bỏ {initial_rows - final_rows} dòng có ngày không hợp lệ")
                
                self.data = self.data.sort_values('date').reset_index(drop=True)
                
        except Exception as e:
            print(f"Lỗi khi xử lý ngày tháng: {e}")

    def _preprocess_and_fit_scalers(self):
        if self.data is None:
            return

        try:
            # Xử lý các giá trị âm và NaN trước khi log transform
            self.data["new_cases"] = self.data["new_cases"].fillna(0).clip(lower=0)
            self.data["new_deaths"] = self.data["new_deaths"].fillna(0).clip(lower=0)
            self.data["new_vaccinations_smoothed"] = self.data["new_vaccinations_smoothed"].fillna(0).clip(lower=0)
            
            # Log transform
            self.data["new_cases_log"] = np.log1p(self.data["new_cases"])
            self.data["new_deaths_log"] = np.log1p(self.data["new_deaths"])
            self.data["vaccinations_log"] = np.log1p(self.data["new_vaccinations_smoothed"])

            # Xử lý các features khác
            self.data["stringency_index"] = self.data["stringency_index"].fillna(0).clip(0, 100)
            self.data["people_fully_vaccinated_per_hundred"] = self.data["people_fully_vaccinated_per_hundred"].fillna(0).clip(lower=0)

            # Fit scalers
            if not self.data["stringency_index"].isna().all():
                self.scalers["stringency_index"] = MinMaxScaler().fit(self.data[["stringency_index"]])
            
            if not self.data["people_fully_vaccinated_per_hundred"].isna().all():
                self.scalers["people_fully_vaccinated_per_hundred"] = RobustScaler().fit(
                    self.data[["people_fully_vaccinated_per_hundred"]]
                )
                
            print("Scalers đã được fit thành công")
            
        except Exception as e:
            print(f"Lỗi khi preprocessing dữ liệu: {e}")

    def _scale_features(self, df):
        df_scaled = df.copy()
        
        try:
            if "stringency_index" in self.scalers and "stringency_index" in df.columns:
                df_scaled["stringency_scaled"] = self.scalers["stringency_index"].transform(df[["stringency_index"]])
            else:
                df_scaled["stringency_scaled"] = 0
                
            if "people_fully_vaccinated_per_hundred" in self.scalers and "people_fully_vaccinated_per_hundred" in df.columns:
                df_scaled["vaccinated_scaled"] = self.scalers["people_fully_vaccinated_per_hundred"].transform(
                    df[["people_fully_vaccinated_per_hundred"]]
                )
            else:
                df_scaled["vaccinated_scaled"] = 0
                
        except Exception as e:
            print(f"Lỗi khi scale features: {e}")
            df_scaled["stringency_scaled"] = 0
            df_scaled["vaccinated_scaled"] = 0
            
        return df_scaled

    def _inverse_scale_new_cases(self, scaled_value):
        try:
            return np.expm1(np.maximum(scaled_value, 0))  # Đảm bảo không âm
        except:
            return max(0, scaled_value)

    def get_latest_data_date(self, country):
        """Lấy ngày dữ liệu mới nhất cho quốc gia"""
        if self.data is None:
            return None
            
        try:
            country_data = self.data[self.data["location"].str.lower() == country.lower()]
            if country_data.empty:
                return None
            return country_data["date"].max().date()
        except Exception as e:
            print(f"Lỗi khi lấy ngày mới nhất: {e}")
            return None

    def get_actual_data(self, country, target_date):
        """Lấy dữ liệu thực tế cho một ngày cụ thể"""
        if self.data is None:
            return None
            
        try:
            country_data = self.data[self.data["location"].str.lower() == country.lower()]
            if country_data.empty:
                return None
                
            if isinstance(target_date, date):
                target_date_dt = datetime.combine(target_date, datetime.min.time())
            else:
                target_date_dt = pd.to_datetime(target_date)
            
            actual_data = country_data[country_data["date"].dt.date == target_date_dt.date()]
            if not actual_data.empty:
                return actual_data.iloc[0]["new_cases"]
        except Exception as e:
            print(f"Lỗi khi lấy dữ liệu thực tế: {e}")
            
        return None

    def prepare_sequence_data(self, country, start_date, days_back=7):
        """Chuẩn bị dữ liệu sequence cho model"""
        try:
            if not self.country_mapper:
                return None, "Country mapper chưa được khởi tạo"
                
            country_id = self.country_mapper.get_country_id(country)
            if country_id is None:
                suggestions = self.country_mapper.suggest_alternatives(country)
                return None, f"Quốc gia '{country}' không được hỗ trợ. Gợi ý: {', '.join(suggestions[:3])}"

            country_data = self.data[self.data["location"].str.lower() == country.lower()].copy()
            if country_data.empty:
                return None, f"Không tìm thấy dữ liệu cho {country}"
                
            country_data = country_data.sort_values("date")

            # Tính ngày kết thúc cho sequence
            if isinstance(start_date, date):
                start_date = datetime.combine(start_date, datetime.min.time())
            end_date_for_sequence = start_date - timedelta(days=1)
            
            relevant_data = country_data[country_data["date"].dt.date <= end_date_for_sequence.date()].tail(days_back)

            if len(relevant_data) < days_back:
                return None, f"Không đủ dữ liệu lịch sử cho {country} đến ngày {end_date_for_sequence.strftime('%d/%m/%Y')} (cần ít nhất {days_back} ngày, chỉ có {len(relevant_data)} ngày)"

            # Scale features
            relevant_data_scaled = self._scale_features(relevant_data.copy())

            # Định nghĩa features cần thiết
            features = [
                'new_cases_log', 'new_deaths_log', 'vaccinations_log',
                'vaccinated_scaled', 'stringency_scaled'
            ]

            # Đảm bảo tất cả features đều có
            for feature in features:
                if feature not in relevant_data_scaled.columns:
                    relevant_data_scaled[feature] = 0

            sequence_data = relevant_data_scaled[features].fillna(0).values
            country_encoded = np.array([country_id])

            return (sequence_data, country_encoded), None

        except Exception as e:
            return None, f"Lỗi khi chuẩn bị dữ liệu: {e}"

    def _parse_target_date(self, target_date):
        """Parse target_date từ nhiều format khác nhau"""
        if isinstance(target_date, date):
            return target_date
        elif isinstance(target_date, datetime):
            return target_date.date()
        elif isinstance(target_date, str):
            try:
                # Thử format DD-MM-YYYY trước
                return datetime.strptime(target_date, '%d-%m-%Y').date()
            except ValueError:
                try:
                    # Thử format YYYY-MM-DD
                    return datetime.strptime(target_date, '%Y-%m-%d').date()
                except ValueError:
                    try:
                        # Thử format DD/MM/YYYY
                        return datetime.strptime(target_date, '%d/%m/%Y').date()
                    except ValueError:
                        # Fallback to today
                        return datetime.today().date()
        elif isinstance(target_date, (int, float)):
            return datetime.today().date() + timedelta(days=int(target_date))
        else:
            return datetime.today().date()

    def predict_cases(self, country, target_date=None, days_ahead=3):
        """Dự đoán số ca nhiễm mới"""
        target_date = self._parse_target_date(target_date)

        try:
            if self.model is None or self.data is None:
                return None, "Model hoặc dữ liệu chưa được tải"

            predictions = {}
            current_date = target_date - timedelta(days=1)

            # Validate country trước khi bắt đầu prediction
            if not self.country_mapper or self.country_mapper.get_country_id(country) is None:
                return None, f"Quốc gia '{country}' không được hỗ trợ"

            for day in range(days_ahead):
                current_date += timedelta(days=1)
                input_data, error = self.prepare_sequence_data(country, current_date, days_back=7)
                
                if error:
                    return None, error

                sequence_data, country_encoded = input_data
                
                sequence_input = sequence_data.reshape(1, 7, len(sequence_data[0]))
                country_input = country_encoded.reshape(1, 1)

                # Predict
                pred_scaled = self.model.predict([sequence_input, country_input], verbose=0)[0][0]
                predicted_value = self._inverse_scale_new_cases(pred_scaled)

                predictions[current_date] = predicted_value

                # Cập nhật sequence cho prediction tiếp theo
                if day < days_ahead - 1:  # Không cần cập nhật ở lần cuối
                    new_row_for_sequence = sequence_data[-1].copy()
                    new_row_for_sequence[0] = pred_scaled
                    sequence_data = np.vstack((sequence_data[1:], new_row_for_sequence))

            return predictions, None

        except Exception as e:
            return None, f"Lỗi khi dự đoán: {e}"

    def get_prediction_confidence(self, country, target_date):
        """Đánh giá độ tin cậy của dự đoán"""
        try:
            latest_data_date = self.get_latest_data_date(country)
            if not latest_data_date:
                return "Không thể đánh giá", "Không có dữ liệu lịch sử để đánh giá độ tin cậy."
                
            if target_date <= latest_data_date:
                return "Cao", f"Dữ liệu thực tế có sẵn cho ngày {target_date.strftime('%d/%m/%Y')}.\nDữ liệu mới nhất: {latest_data_date.strftime('%d/%m/%Y')}"

            data_to_target_gap_days = (target_date - latest_data_date).days
            
            if data_to_target_gap_days <= 3:
                return "Cao", f"Dự đoán có độ tin cậy cao (cách dữ liệu mới nhất {data_to_target_gap_days} ngày).\nDữ liệu mới nhất: {latest_data_date.strftime('%d/%m/%Y')}"
            elif data_to_target_gap_days <= 7:
                return "Trung bình", f"Dự đoán có độ tin cậy trung bình (cách dữ liệu mới nhất {data_to_target_gap_days} ngày).\nDữ liệu mới nhất: {latest_data_date.strftime('%d/%m/%Y')}"
            elif data_to_target_gap_days <= 14:
                return "Thấp", f"Dự đoán có độ tin cậy thấp (cách dữ liệu mới nhất {data_to_target_gap_days} ngày).\nDữ liệu mới nhất: {latest_data_date.strftime('%d/%m/%Y')}"
            else:
                return "Rất thấp", f"CẢNH BÁO: Dự đoán có độ tin cậy rất thấp (cách dữ liệu mới nhất {data_to_target_gap_days} ngày).\nDữ liệu mới nhất: {latest_data_date.strftime('%d/%m/%Y')}"
                
        except Exception as e:
            return "Lỗi", f"Không thể đánh giá độ tin cậy: {e}"

    def get_available_countries(self):
        """Lấy danh sách các quốc gia có sẵn"""
        if self.country_mapper:
            return self.country_mapper.get_supported_countries()
        return []

    def find_country_from_text(self, text):
        """Tìm quốc gia từ text"""
        if self.country_mapper:
            return self.country_mapper.find_country(text)
        return None

    def format_prediction_response(self, country, predictions, confidence_level, confidence_msg):
        """Format response cho prediction"""
        if not predictions:
            return "Không thể thực hiện dự đoán."

        response = f"Dự đoán COVID-19 cho {country}\n\n"
        
        for pred_date, pred_value in predictions.items():
            actual_value = self.get_actual_data(country, pred_date)
            response += f"{pred_date.strftime('%d/%m/%Y')}: {pred_value:.0f} ca mới"
            
            if actual_value is not None:
                diff = abs(actual_value - pred_value)
                accuracy = max(0, 100 - (diff / max(actual_value, 1) * 100))
                response += f" (Thực tế: {actual_value:.0f}, Độ chính xác: {accuracy:.1f}%)"
            response += "\n"
        
        response += f"\nĐộ tin cậy: {confidence_level}\n{confidence_msg}\n"
        response += f"\nDự đoán được thực hiện lúc: {datetime.now().strftime('%H:%M %d/%m/%Y')}"
        
        return response

@st.cache_resource
def get_prediction_service():
    return CovidPredictionService()