from PIL import Image, ImageDraw, ImageFont

# Load both images
img1 = Image.open("output_dog.png")
img2 = Image.open("output_cyberpunk.png")

# Add some spacing and a label strip below each image
label_height = 40
gap = 10

width = img1.width + img2.width + gap
height = max(img1.height, img2.height) + label_height

combined = Image.new("RGB", (width, height), "white")

# Paste the two images side by side
combined.paste(img1, (0, 0))
combined.paste(img2, (img1.width + gap, 0))

# Add labels underneath each image
draw = ImageDraw.Draw(combined)
try:
    font = ImageFont.truetype("arial.ttf", 20)
except:
    font = ImageFont.load_default()

draw.text((img1.width // 2 - 60, img1.height + 5), "Golden Retriever", fill="black", font=font)
draw.text((img1.width + gap + img2.width // 2 - 90, img2.height + 5), "Cyberpunk Street", fill="black", font=font)

combined.save("combined_output.png")
print("Saved combined_output.png")