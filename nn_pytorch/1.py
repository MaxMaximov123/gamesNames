import requests

print(requests.post('https://8c49-178-204-193-130.ngrok-free.app/api/forecast', json={'n1': 'Лила А.-А.', 'n2': 'Andreea A Lila'}).json())