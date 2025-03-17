import base64
from io import BytesIO
import requests

url = 'http://127.0.0.1:5010/api/v1/hc_emci/predict'

payload = {}
headers = {
    'Content-Type': 'application/json'
}

with open('./data/g0.png', 'rb') as f:
    image = base64.b64encode(f.read()).decode('utf-8')

payload['image'] = image

resp = requests.post(url, json=payload, headers=headers, verify=False)
print(resp.text)
print(resp.status_code)
print(resp.json)
