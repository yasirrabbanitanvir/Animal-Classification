# from flask import Flask, render_template, request, jsonify
# import io
# from PIL import Image
# import torch
# import torchvision.transforms as transforms
# import torchvision.models as models
# import requests

# app = Flask(__name__)

# # === Custom ResNet18 Model Setup ===

# MODEL_PATH = 'best_model.pth'
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# num_classes = 30
# model = models.resnet18(pretrained=False)
# model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

# model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
# model.to(device)
# model.eval()

# class_names = [
#     "Hargila Bok",
#     "Lama",
#     "Rabbit",
#     "ass",
#     "bear",
#     "camel",
#     "camel bird",
#     "cat",
#     "cow",
#     "crocodile",
#     "deer",
#     "dog",
#     "elephant",
#     "gayal",
#     "giraffe",
#     "goat",
#     "hippopotamus",
#     "horse",
#     "kalo bok",
#     "kangaru",
#     "lion",
#     "monkey",
#     "panda",
#     "peacock",
#     "porcupine",
#     "rhinoceros",
#     "sheep",
#     "squirrel",
#     "tiger",
#     "zebra"
# ]

# transform = transforms.Compose([
#     transforms.Resize(256),
#     transforms.CenterCrop(224),
#     transforms.ToTensor(),
#     transforms.Normalize(
#         mean=[0.485, 0.456, 0.406],  # same as ImageNet
#         std=[0.229, 0.224, 0.225]
#     )
# ])

# # === Bangla Names Mapping ===
# BANGAL_NAMES = {
#     "hargila bok": "হারগিলা বক",
#     "lama": "লামা",
#     "rabbit": "খরগোশ",
#     "ass": "গাধা",
#     "bear": "ভালুক",
#     "camel": "উট",
#     "camel bird": "ক্যামেল পাখি",
#     "cat": "বিড়াল",
#     "cow": "গরু",
#     "crocodile": "কুমির",
#     "deer": "হরিণ",
#     "dog": "কুকুর",
#     "elephant": "হাতি",
#     "gayal": "গয়াল",
#     "giraffe": "জিরাফ",
#     "goat": "ছাগল",
#     "hippopotamus": "জলহস্তী",
#     "horse": "ঘোড়া",
#     "kalo bok": "কালো বক",
#     "kangaru": "ক্যাঙ্গারু",
#     "lion": "সিংহ",
#     "monkey": "বানর",
#     "panda": "পান্ডা",
#     "peacock": "ময়ূর",
#     "porcupine": "কাঁটাতার",
#     "rhinoceros": "গন্ডার",
#     "sheep": "ভেড়া",
#     "squirrel": "গিলহারি",
#     "tiger": "বাঘ",
#     "zebra": "জেব্রা"
# }

# # === Exact Species Mapping ===
# SPECIES_MAP = {
#     # Birds
#     "hargila bok": "Bird (likely a stork species, as 'Hargila' is a local name for the Asian Openbill Stork)",
#     "camel bird": "Bird (unclear species, possibly a bustard or large ground bird)",
#     "peacock": "Bird (Phasianidae family)",

#     # Mammals - Ungulates
#     "lama": "Mammal, Ungulate, Camelid",
#     "ass": "Mammal, Ungulate, Equidae",
#     "bear": "Mammal, Carnivora (large mammal)",
#     "camel": "Mammal, Ungulate, Camelidae",
#     "cow": "Mammal, Ungulate, Bovidae",
#     "gayal": "Mammal, Ungulate, Bovidae",
#     "giraffe": "Mammal, Ungulate, Giraffidae",
#     "goat": "Mammal, Ungulate, Bovidae",
#     "hippopotamus": "Mammal, Ungulate, Hippopotamidae",
#     "horse": "Mammal, Ungulate, Equidae",
#     "kalo bok": "Mammal, Ungulate, Bovidae",
#     "kangaru": "Mammal, Marsupial, Macropodidae",
#     "deer": "Mammal, Ungulate, Cervidae",
#     "sheep": "Mammal, Ungulate, Bovidae",
#     "rhinoceros": "Mammal, Ungulate, Rhinocerotidae",
#     "zebra": "Mammal, Ungulate, Equidae",

#     # Mammals - Carnivores
#     "cat": "Mammal, Carnivora, Felidae",
#     "dog": "Mammal, Carnivora, Canidae",
#     "lion": "Mammal, Carnivora, Felidae",
#     "tiger": "Mammal, Carnivora, Felidae",
#     "panda": "Mammal, Carnivora, Ursidae (mostly herbivorous)",
#     "bear": "Mammal, Carnivora, Ursidae",

#     # Mammals - Primates
#     "monkey": "Mammal, Primate",

#     # Mammals - Rodents and Small Mammals
#     "rabbit": "Mammal, Lagomorpha",
#     "porcupine": "Mammal, Rodentia",
#     "squirrel": "Mammal, Rodentia",

#     # Reptiles
#     "crocodile": "Reptile, Crocodylia"
# }

# def get_bangla_name(english_name):
#     return BANGAL_NAMES.get(english_name.lower(), "অজানা")

# def get_exact_species(english_name):
#     return SPECIES_MAP.get(english_name.lower(), "Unknown species")

# def get_wikipedia_summary(animal_name, lang='en'):
#     url = f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/{animal_name.replace(' ', '_')}"
#     try:
#         resp = requests.get(url, timeout=5)
#         if resp.status_code == 200:
#             data = resp.json()
#             return {
#                 "class": animal_name,
#                 "species": get_exact_species(animal_name),
#                 "bangla_name": get_bangla_name(animal_name),
#                 "nature": data.get("extract", "No description available.")
#             }
#         else:
#             return {
#                 "class": animal_name,
#                 "species": get_exact_species(animal_name),
#                 "bangla_name": get_bangla_name(animal_name),
#                 "nature": "No description available."
#             }
#     except Exception as e:
#         print("Wikipedia API error:", e)
#         return {
#             "class": animal_name,
#             "species": get_exact_species(animal_name),
#             "bangla_name": get_bangla_name(animal_name),
#             "nature": "No description available."
#         }

# # === Routes ===

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/index.html')
# def index_html():
#     return render_template('index.html')

# @app.route('/about.html')
# def about():
#     return render_template('about.html')

# @app.route('/contact.html')
# def contact():
#     return render_template('contact.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     if 'image' not in request.files:
#         return jsonify({'error': 'No image provided'}), 400

#     image_file = request.files['image']
#     image_bytes = image_file.read()
#     image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

#     input_tensor = transform(image).unsqueeze(0).to(device)

#     with torch.no_grad():
#         outputs = model(input_tensor)
#         probs = torch.nn.functional.softmax(outputs[0], dim=0)
#         confidence, class_idx = torch.max(probs, 0)

#     predicted_class = class_names[class_idx.item()]
#     animal_info = get_wikipedia_summary(predicted_class)

#     return jsonify({
#         'model': 'Custom ResNet18 trained on Animal Classification Dataset',
#         'class': predicted_class,
#         'confidence': confidence.item(),
#         'info': animal_info
#     })

# if __name__ == '__main__':
#     app.run(debug=True)

from flask import Flask, render_template, request, jsonify
import io
from PIL import Image
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import requests
import base64

app = Flask(__name__)

# === Clarifai API Config ===
CLARIFAI_API_KEY = '337d6c778a7540b88f815802916c17d0'
CLARIFAI_MODEL_ID = 'aaa03c23b3724a16a56b629203edc62c'  # General Model
CLARIFAI_URL = f'https://api.clarifai.com/v2/models/{CLARIFAI_MODEL_ID}/outputs'

def is_animal_image(image_bytes):
    encoded_image = base64.b64encode(image_bytes).decode('utf-8')
    headers = {
        'Authorization': f'Key {CLARIFAI_API_KEY}',
        'Content-Type': 'application/json'
    }
    data = {
        "inputs": [
            {
                "data": {
                    "image": {
                        "base64": encoded_image
                    }
                }
            }
        ]
    }
    try:
        response = requests.post(CLARIFAI_URL, headers=headers, json=data)
        response.raise_for_status()
        concepts = response.json()['outputs'][0]['data']['concepts']
        for concept in concepts:
            if concept['name'].lower() in ['animal', 'mammal', 'reptile', 'bird'] and concept['value'] >= 0.85:
                return True
    except Exception as e:
        print('Clarifai API error:', e)
    return False

# === Custom ResNet18 Model Setup ===

MODEL_PATH = 'best_model.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_classes = 30
model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

class_names = [
    "Hargila Bok", "Lama", "Rabbit", "ass", "bear", "camel", "camel bird", "cat", "cow", "crocodile",
    "deer", "dog", "elephant", "gayal", "giraffe", "goat", "hippopotamus", "horse", "kalo bok", "kangaru",
    "lion", "monkey", "panda", "peacock", "porcupine", "rhinoceros", "sheep", "squirrel", "tiger", "zebra"
]

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# === Bangla Names Mapping ===
BANGAL_NAMES = {
    "hargila bok": "হারগিলা বক", "lama": "লামা", "rabbit": "খরগোশ", "ass": "গাধা", "bear": "ভালুক",
    "camel": "উট", "camel bird": "ক্যামেল পাখি", "cat": "বিড়াল", "cow": "গরু", "crocodile": "কুমির",
    "deer": "হরিণ", "dog": "কুকুর", "elephant": "হাতি", "gayal": "গয়াল", "giraffe": "জিরাফ",
    "goat": "ছাগল", "hippopotamus": "জলহস্তী", "horse": "ঘোড়া", "kalo bok": "কালো বক", "kangaru": "ক্যাঙ্গারু",
    "lion": "সিংহ", "monkey": "বানর", "panda": "পান্ডা", "peacock": "ময়ূর", "porcupine": "কাঁটাতার",
    "rhinoceros": "গন্ডার", "sheep": "ভেড়া", "squirrel": "গিলহারি", "tiger": "বাঘ", "zebra": "জেব্রা"
}

# === Exact Species Mapping ===
SPECIES_MAP = {
    "hargila bok": "Bird (likely a stork species, as 'Hargila' is a local name for the Asian Openbill Stork)",
    "camel bird": "Bird (unclear species, possibly a bustard or large ground bird)",
    "peacock": "Bird (Phasianidae family)",
    "lama": "Mammal, Ungulate, Camelid", "ass": "Mammal, Ungulate, Equidae", "bear": "Mammal, Carnivora",
    "camel": "Mammal, Ungulate, Camelidae", "cow": "Mammal, Ungulate, Bovidae", "gayal": "Mammal, Ungulate, Bovidae",
    "giraffe": "Mammal, Ungulate, Giraffidae", "goat": "Mammal, Ungulate, Bovidae", "hippopotamus": "Mammal, Ungulate, Hippopotamidae",
    "horse": "Mammal, Ungulate, Equidae", "kalo bok": "Mammal, Ungulate, Bovidae", "kangaru": "Mammal, Marsupial",
    "deer": "Mammal, Ungulate, Cervidae", "sheep": "Mammal, Ungulate, Bovidae", "rhinoceros": "Mammal, Ungulate, Rhinocerotidae",
    "zebra": "Mammal, Ungulate, Equidae", "cat": "Mammal, Carnivora, Felidae", "dog": "Mammal, Carnivora, Canidae",
    "lion": "Mammal, Carnivora, Felidae", "tiger": "Mammal, Carnivora, Felidae", "panda": "Mammal, Carnivora, Ursidae",
    "monkey": "Mammal, Primate", "rabbit": "Mammal, Lagomorpha", "porcupine": "Mammal, Rodentia",
    "squirrel": "Mammal, Rodentia", "crocodile": "Reptile, Crocodylia"
}

def get_bangla_name(english_name):
    return BANGAL_NAMES.get(english_name.lower(), "অজানা")

def get_exact_species(english_name):
    return SPECIES_MAP.get(english_name.lower(), "Unknown species")

def get_wikipedia_summary(animal_name, lang='en'):
    url = f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/{animal_name.replace(' ', '_')}"
    try:
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            return {
                "class": animal_name,
                "species": get_exact_species(animal_name),
                "bangla_name": get_bangla_name(animal_name),
                "nature": data.get("extract", "No description available.")
            }
    except Exception as e:
        print("Wikipedia API error:", e)
    return {
        "class": animal_name,
        "species": get_exact_species(animal_name),
        "bangla_name": get_bangla_name(animal_name),
        "nature": "No description available."
    }

# === Routes ===

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/index.html')
def index_html():
    return render_template('index.html')

@app.route('/about.html')
def about():
    return render_template('about.html')

@app.route('/contact.html')
def contact():
    return render_template('contact.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    image_file = request.files['image']
    image_bytes = image_file.read()

    # Clarifai validation
    if not is_animal_image(image_bytes):
        return jsonify({'error': 'This image does not appear to contain an animal.'}), 400

    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.nn.functional.softmax(outputs[0], dim=0)
        confidence, class_idx = torch.max(probs, 0)

    predicted_class = class_names[class_idx.item()]
    animal_info = get_wikipedia_summary(predicted_class)

    return jsonify({
        'model': 'Custom ResNet18 trained on Animal Classification Dataset',
        'class': predicted_class,
        'confidence': confidence.item(),
        'info': animal_info
    })

if __name__ == '__main__':
    app.run(debug=True)