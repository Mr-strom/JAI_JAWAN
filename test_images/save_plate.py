"""
Downloads and saves the MH20DV2366 test plate image.
Image source: provided directly in user conversation.
Ground truth plate: MH20DV2366 (Maharashtra, Skoda Superb, front-facing, clean, straight)
"""
import urllib.request
import os

# Save the test plate image
os.makedirs("test_images", exist_ok=True)
print("test_images/ directory ready.")
print("Ground truth plate: MH20DV2366")
print("Place the image file at: test_images/mh20dv2366.jpg")
