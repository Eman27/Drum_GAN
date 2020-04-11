import requests

url = 'http://localhost:5000/results'
r = requests.post(url,json={'genre':'general'})

print(r.json())