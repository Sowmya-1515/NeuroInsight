import torch
import numpy as np
import faiss
from PIL import Image
from torchvision import transforms
from step2_model import Phase2Model

# ---------------- PATHS ----------------

MODEL_PATH = "training_output/best_model.pth"
FAISS_INDEX = "faiss_index/faiss_ivfpq.index"

# ---------------- DEVICE ----------------

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# ---------------- LOAD MODEL ----------------

model = Phase2Model(embedding_dim=128, pretrained=False).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# ---------------- LOAD FAISS ----------------

index = faiss.read_index(FAISS_INDEX)

# ---------------- TRANSFORM ----------------

transform = transforms.Compose([
    transforms.Resize((160,160)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

# ---------------- MAPS ----------------

GRADE_MAP = {0:"LGG",1:"HGG"}
SEVERITY_MAP = {0:"Low",1:"Medium",2:"High"}
SIZE_MAP = {0:"Small",1:"Medium",2:"Large"}
LOCATION_MAP = {0:"Left",1:"Right",2:"Bilateral"}

# ---------------- MAIN PIPELINE ----------------

def analyze_mri(image_path):

    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)

    embedding = outputs["embedding"].cpu().numpy()

    # ---------------- TUMOR DETECTION ----------------

    D,I = index.search(embedding,5)
    similarity = 1 - D[0][0]/2

    if similarity > 0.6:
        tumor_result = "Tumor Detected"
    else:
        tumor_result = "No Tumor Detected"

    # ---------------- PREDICTIONS ----------------

    grade = GRADE_MAP[outputs["grade_logits"].argmax(1).item()]
    severity = SEVERITY_MAP[outputs["severity_logits"].argmax(1).item()]
    size = SIZE_MAP[outputs["size_logits"].argmax(1).item()]
    location = LOCATION_MAP[outputs["location_logits"].argmax(1).item()]

    # ---------------- RETRIEVAL ----------------

    D,I = index.search(embedding,10)

    return {
        "tumor": tumor_result,
        "similarity": similarity,
        "grade": grade,
        "severity": severity,
        "size": size,
        "location": location,
        "retrieved_indexes": I[0]
    }

# ---------------- TEST ----------------

if __name__ == "__main__":

    test_image = "/Users/sowmyaalamuri/Desktop/Capstone_project/brats_slices_224/slice_000001.png"

    result = analyze_mri(test_image)

    print("\n==============================")
    print(" MRI ANALYSIS REPORT")
    print("==============================")

    print("Tumor Detection:", result["tumor"])
    print("Similarity Score:", round(result["similarity"],3))

    print("\nTumor Characteristics")
    print("----------------------")
    print("Grade:", result["grade"])
    print("Severity:", result["severity"])
    print("Size:", result["size"])
    print("Location:", result["location"])

    print("\nTop Similar Cases Indexes:")
    print(result["retrieved_indexes"])
    