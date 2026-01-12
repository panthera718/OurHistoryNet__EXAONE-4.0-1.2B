import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from collections import deque
from tqdm import tqdm
import json
import time

##START_URL = "https://contents.history.go.kr/front/nh/main.do"
START_URL = "https://contents.history.go.kr/front/nh/view.do?levelId=nh_009_0020"
DOMAIN = "contents.history.go.kr"
OUTPUT_PATH = "nh_raw.jsonl"

MAX_PAGES = 50  # Safety device: Avoid scratching too much

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; EXAONE-StudyBot/1.0)"
}

def clean_text(text: str) -> str:
    # Simple function for cleaning up whitespace
    lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln]  # Remove blank lines
    return "\n".join(lines)

def extract_text(url, soup: BeautifulSoup) -> str:
    """
    A function that extracts the body text from a page.
    Since I don't know the exact structure of the site, I'll focus on the p tag first.
    If necessary, you can narrow it down to div.content-body, etc.
    """
    # 1. p tag
    paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
    text = "\n".join(paragraphs)

    text = clean_text(text)
    return text

def crawl():
    visited = set()
    q = deque([START_URL])

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f_out:
        pbar = tqdm(total=MAX_PAGES, desc="Crawling nh")
        while q and len(visited) < MAX_PAGES:
            url = q.popleft()
            if url in visited:
                continue
            visited.add(url)

            try:
                resp = requests.get(url, headers=HEADERS, timeout=10)
                resp.raise_for_status()
            except Exception as e:
                print(f"[WARN] Request failed: {url} ({e})")
                continue

            soup = BeautifulSoup(resp.text, "html.parser")

            # title
            title_tag = soup.find("title")
            title = title_tag.get_text(strip=True) if title_tag else url

            text = extract_text(url, soup)
            if text.strip():
                record = {"url": url, "title": title, "text": text}
                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                pbar.set_postfix_str("saved")
            else:
                pbar.set_postfix_str("no-text")

            # Link navigation: restricted to same domain + nh area
            for a in soup.find_all("a", href=True):
                href = a["href"].strip()
                next_url = urljoin(url, href)
                pr = urlparse(next_url)

                if pr.netloc != DOMAIN:
                    continue
                # Only the new Korean history section (conditions can be added if necessary)
                if "/nh/" not in pr.path:
                    continue

                if next_url not in visited:
                    q.append(next_url)

            pbar.update(1)
            time.sleep(0.4)  # Delay to avoid overloading the server

        pbar.close()

    print(f"[DONE] Total {len(visited)} pages visited, results: {OUTPUT_PATH}")

if __name__ == "__main__":
    crawl()
