import os
import time
import base64
import requests
from io import BytesIO
from PIL import Image
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

# ---------- SETTINGS ----------
search_keyword = "RTC MODULE HW-084"
target_size = (640, 640)
num_scrolls = 3
output_dir = f"images/{search_keyword.replace(' ', '_')}_selenium"
os.makedirs(output_dir, exist_ok=True)

# ---------- SETUP CHROME DRIVER ----------
chrome_options = Options()
# chrome_options.add_argument("--headless")  # Keep commented for debugging
chrome_options.add_argument("--disable-blink-features=AutomationControlled")
chrome_options.add_argument("--start-maximized")

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)

# ---------- SEARCH GOOGLE IMAGES ----------
query = search_keyword.replace(' ', '+')
url = f"https://www.google.com/search?tbm=isch&q={query}"
driver.get(url)
time.sleep(3)

# ---------- SCROLL TO LOAD MORE ----------
for _ in range(num_scrolls):
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(2)

# ---------- COLLECT IMAGE ELEMENTS ----------
image_elements = driver.find_elements(By.CSS_SELECTOR, "img")

# ---------- FUNCTION TO RESIZE + PAD ----------
def resize_and_pad(img, size=(640, 640), color=(255, 255, 255)):
    img.thumbnail(size, Image.ANTIALIAS)
    new_img = Image.new("RGB", size, color)
    new_img.paste(
        img, ((size[0] - img.size[0]) // 2, (size[1] - img.size[1]) // 2)
    )
    return new_img

# ---------- DOWNLOAD IMAGES ----------
count = 0
for idx, img_el in enumerate(image_elements):
    try:
        img_url = img_el.get_attribute("src") or img_el.get_attribute("data-src")

        if not img_url:
            continue

        if img_url.startswith("data:image"):
            header, base64_data = img_url.split(",", 1)
            image_data = base64.b64decode(base64_data)
            img = Image.open(BytesIO(image_data)).convert("RGB")
        elif img_url.startswith("http"):
            response = requests.get(img_url, timeout=10)
            img = Image.open(BytesIO(response.content)).convert("RGB")
        else:
            continue

        img = resize_and_pad(img, target_size)
        img.save(os.path.join(output_dir, f"{count:04}.jpg"))
        count += 1

    except Exception as e:
        print(f"⚠️ Skipping image {idx}: {e}")

# ---------- DONE ----------
driver.quit()
print(f"✅ Saved {count} resized images to {output_dir}")
