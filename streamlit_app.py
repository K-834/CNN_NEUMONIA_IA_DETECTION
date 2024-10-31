import streamlit as st
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image

# Cargar el modelo completo desde el archivo
model_path = 'ResNet18_COMPLETO_best_12.pt'  # Asegúrate de que este sea el path correcto
try:
    model = torch.load(model_path, map_location=torch.device('cpu'))  # Cargar el modelo completo
except RuntimeError as e:
    st.error(f"Error al cargar el modelo: {e}")

model.eval()

# Definir las transformaciones de las imágenes
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Cambiar a (224, 224) para asegurar dimensiones correctas
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Función para preprocesar la imagen
def preprocess_image(img):
    img = img.convert('RGB')  # Asegurarse de que la imagen esté en RGB
    img_tensor = transform(img)
    img_tensor = img_tensor.unsqueeze(0)  # Agregar una dimensión para el batch
    return img_tensor

# Función para hacer una predicción
def predict(img):
    img_tensor = preprocess_image(img)
    with torch.no_grad():
        output = model(img_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)  # Obtener las probabilidades
        predicted = torch.argmax(probabilities, dim=1)  # Obtener la clase con mayor probabilidad
        confidence = probabilities[0][predicted].item()  # Obtener la confianza en la predicción
        return "Neumonía" if predicted >= 0.8 else "Normal", confidence * 100  # Retornar la clase y la confianza

# Interfaz de usuario con Streamlit
def main():
    st.title("Clasificador de Neumonía con ResNet18")
    uploaded_file = st.file_uploader("Subir una imagen", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Imagen subida', use_column_width=True)
        prediction, confidence = predict(image)
        st.write("Predicción:", prediction)
        st.write("Confianza de la predicción: {:.4f}%".format(confidence))

if __name__ == "__main__":
    main()
