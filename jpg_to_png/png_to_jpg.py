from pathlib import Path
from PIL import Image

inputPath = Path(r"fire_dataset\fire_images")
inputFiles = inputPath.glob("**/*.png")
outputPath = Path(r"fire_dataset\fire_images1")
for f in inputFiles:
    outputFile = outputPath / Path(f.stem + ".jpg")
    im = Image.open(f)
    im.save(outputFile)
