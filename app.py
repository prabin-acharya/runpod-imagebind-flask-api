from flask import Flask, request, jsonify
from imagebind import data
import torch
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType

app = Flask(__name__)

# Instantiate model
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(device)
print("#####################################")
model = imagebind_model.imagebind_huge(pretrained=True)
model.eval()
model.to(device)

@app.route('/embeddings', methods=['GET'])
def get_embeddings():
    # Get text input from request
    text_list = ["A dog.", "A car", "A bird"]

    # Load and transform text data
    text_input = data.load_and_transform_text(text_list, device)

    with torch.no_grad():
        # Generate embeddings
        embeddings = model({ModalityType.TEXT: text_input})

    print(embeddings)
    # Convert embeddings to JSON format and return
    embeddings_json = embeddings[ModalityType.TEXT].cpu().numpy().tolist()  # Assuming TEXT modality
    return jsonify(embeddings=embeddings_json)

if __name__ == '__main__':
    app.run(host='0.0.0.0')  # Run Flask app on port 8000


