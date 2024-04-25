import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
from resnet_model import CustomResNet18  # Import your model architecture
from tempplan import preprocess_image  # Import your image preprocessing function

# Load your trained model
model = CustomResNet18(num_classes=1)  # Example: CustomResNet18 is your model architecture
model.load_state_dict(torch.load(r"C:\Users\ashir\OneDrive\Documents\MY CODE\model.pth"))
model.eval()

# Define image preprocessing transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Define the Streamlit app
def main():
    st.title('Deepfake Detection App')
    uploaded_image = st.file_uploader('Upload an image', type=['jpg', 'png', 'jpeg'])

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Preprocess the image
        processed_image = preprocess_image(image, transform)
        prediction = model(processed_image.unsqueeze(0)).item()

        st.write('Prediction:', prediction)

if __name__ == '__main__':
    main()
