import requests
from PIL import Image
from io import BytesIO
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from caption_generator import generate_caption_for_pil_image

HEADERS = {"User-Agent": "Mozilla/5.0"}

def caption_images_from_url(url: str, min_pixels: int = 200, timeout: int = 10) -> dict:
    """
    Scrapes all <img> tags from the given URL, generates a caption for each
    valid image, and returns a dict mapping image URL -> caption.

    Args:
        url: The page URL to scrape images from.
        min_pixels: Minimum total pixel count (width * height) to keep an image.
        timeout: Timeout in seconds for image requests.

    Returns:
        Dict of {image_url: caption}.
    """
    captions: dict[str, str] = {}

    response = requests.get(url, headers=HEADERS)
    print("Status:", response.status_code, "HTML size:", len(response.text))

    soup = BeautifulSoup(response.text, "html.parser")
    img_elements = soup.find_all("img")
    print(f"Found {len(img_elements)} <img> tags")

    for idx, img_element in enumerate(img_elements, start=1):
        img_url = img_element.get("src") or img_element.get("data-src")
        if not img_url and img_element.has_attr("srcset"):
            img_url = img_element["srcset"].split()[0]
        if not img_url:
            continue

        # Skip SVGs directly
        if ".svg" in img_url:
            continue

        # Resolve relative URLs against the source page
        img_url = urljoin(url, img_url)
        if not img_url.startswith("http"):
            continue

        try:
            r = requests.get(img_url, timeout=timeout, headers=HEADERS)
            raw_image = Image.open(BytesIO(r.content))

            # Skip very small images
            if raw_image.size[0] * raw_image.size[1] < min_pixels:
                continue

            raw_image = raw_image.convert("RGB")

            caption = generate_caption_for_pil_image(raw_image)
            captions[img_url] = caption

        except OSError:
            # Skip images PIL cannot open (SVG, ICO, corrupt files)
            continue
        except Exception as e:
            print(f"[{idx}] Error: {e}")
            continue

    return captions
