from PIL import Image, ImageDraw, ImageFont
import os

# Create a new image with a white background
width = 1200
height = 800
image = Image.new('RGB', (width, height), 'white')
draw = ImageDraw.Draw(image)

# Try to load a monospace font, fall back to default if not available
try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 14)
except:
    font = ImageFont.load_default()

# Read the ASCII art
with open('docs/images/architecture.txt', 'r') as f:
    ascii_art = f.read()

# Calculate text size and position
lines = ascii_art.split('\n')
line_height = 20
text_width = max(len(line) * 8 for line in lines)
text_height = len(lines) * line_height

# Calculate starting position to center the text
x = (width - text_width) // 2
y = (height - text_height) // 2

# Draw each line of text
for i, line in enumerate(lines):
    draw.text((x, y + i * line_height), line, font=font, fill='black')

# Save the image
image.save('docs/images/architecture.png') 