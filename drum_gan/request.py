import requests

url = 'http://localhost:5000/train'
r = requests.post(url,json={'genres':'rock'})

print(r.json())