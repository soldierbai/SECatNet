from flask import Flask, request, jsonify
from flask_cors import CORS
import base64  # 新增base64模块
from io import BytesIO  # 新增字节流处理
from werkzeug.utils import secure_filename
import os
import torch
from PIL import Image
import io
import argparse
from src.secatnet import SECatNet
from torchvision import transforms


model = None
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.11932160705327988], std=[0.2542687952518463])
])


def load_model(model_path, device):
    global model
    model = SECatNet(num_classes=2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    model.to(device)


app = Flask(__name__)
CORS(app, supports_credentials=True)

UPLOAD_FOLDER = 'temp_uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/api/v1/hc_emci/predict', methods=['POST'])
def predict():
    """接收base64图片并返回预测结果"""
    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({'error': '缺少base64图像数据'}), 400

    try:
        base64_str = data['image'].split(',')[-1]
        img_bytes = base64.b64decode(base64_str)

        image = Image.open(BytesIO(img_bytes)).convert('L')
        tensor = transform(image).unsqueeze(0).to(args.device)

        with torch.no_grad():
            output = model(tensor)
            print(output)
            probabilities = torch.softmax(output, dim=1).cpu().numpy().flatten().tolist()

        return jsonify({
            'probabilities': probabilities,
            'class_0': float(output[0][0]),
            'class_1': float(output[0][1])
        })
    except (base64.binascii.Error, ValueError) as e:
        return jsonify({'error': f'Base64解码失败: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'error': f'图像处理失败: {str(e)}'}), 500
    finally:
        if 'img_bytes' in locals():
            del img_bytes


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='模型路径')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    load_model(args.model, args.device)
    app.run(host='0.0.0.0', port=5010, threaded=True)