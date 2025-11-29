import sys
import os
from pathlib import Path
from tensorflow.keras.models import load_model
from tensorflow.keras import layers

# Thêm đường dẫn để import modules
sys.path.append("Web")

def check_model_parameters():
    """Kiểm tra thông số của model đã train."""
    print("Kiểm Tra Thông Số Model BiLSTM Sau Khi Train")
    print("=" * 60)
    
    try:
        # Đường dẫn tới model
        model_path = Path("Web/data/bilstm_covid19_model_with_emb.h5")
        
        if not model_path.exists():
            print(f"Lỗi: Không tìm thấy file model tại {model_path}")
            print("Vui lòng đảm bảo model đã được huấn luyện và lưu đúng đường dẫn.")
            return
        
        print(f"Đang tải model từ: {model_path}")
        model = load_model(str(model_path))
        print("Model đã được tải thành công.")
        
        print("\nCấu trúc tổng quan của model:")
        print("-" * 30)
        model.summary()
        
        print("\nChi tiết về các layers, đặc biệt là Embedding Layer:")
        print("-" * 30)
        found_embedding_layer = False
        for i, layer in enumerate(model.layers):
            print(f"Layer {i}: {layer.name} - {type(layer).__name__}")
            if "embedding" in layer.name.lower() and isinstance(layer, layers.Embedding):
                found_embedding_layer = True
                input_dim = getattr(layer, 'input_dim', 'N/A')
                output_dim = getattr(layer, 'output_dim', 'N/A')
                print(f"  EMBEDDING LAYER (Quốc gia) ĐƯỢC TÌM THẤY!")
                print(f"  Input Dimension (Số lượng quốc gia hỗ trợ): {input_dim}")
                print(f"  Output Dimension (Kích thước embedding): {output_dim}")
                
                # Lấy weights của embedding layer để xác nhận kích thước
                weights = layer.get_weights()
                if weights:
                    embedding_matrix = weights[0]
                    print(f"  Kích thước ma trận Embedding: {embedding_matrix.shape}")
                    print(f"  XÁC NHẬN: Model này hỗ trợ {embedding_matrix.shape[0]} quốc gia.")
            print()
        
        if not found_embedding_layer:
            print("Không tìm thấy Embedding Layer cho quốc gia trong model này.")
            
        print("\nKiểm tra thông số model hoàn tất.")
        
    except Exception as e:
        print(f"Lỗi khi kiểm tra thông số model: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_model_parameters()