import os
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()


def download_audio(yt_id: str, folder: Path, duration: float) -> None:
    proxy_url = os.getenv("AUDIO_FETCH_URL", "http://localhost:5000/fetch_audio")
    output_path = folder / f"{yt_id}.m4a"

    if proxy_url:
        params = {"yt_id": yt_id, "duration": duration}

        response = requests.get(proxy_url, params=params, stream=True, timeout=120)
        if response.status_code != 200:
            raise RuntimeError(
                f"Proxy fetch failed with {response.status_code}: {response.text}"
            )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        return
