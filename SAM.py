from ultralytics import SAM

# Load a model
model = SAM('sam_b.pt')

# Display model information (optional)
model.info()

# Run inference
hamed=model(r"C:\Users\VAIO\Desktop\imagenet_21k\tu1.jpg")

print(hamed)