import json
import logging
import os

from dotenv import load_dotenv

from config.config import TABLE

load_dotenv()

logger = logging.getLogger("experiment")


def main():
    dataset = []
    for genre in os.listdir("dataset/audio/"):
        if genre not in TABLE:
            continue
        genre_path = os.path.join("dataset/audio/", genre)
        if os.path.isdir(genre_path):
            for title in os.listdir(genre_path):
                title_path = os.path.join(genre_path, title)
                if os.path.isfile(title_path):
                    dataset.append(
                        {
                            "yt_id": title.split(".")[0],
                            "genres": genre,
                            "audio_path": title_path,
                        }
                    )
    with open("dataset/p2_dataset.json", "w") as f:
        json.dump(dataset, f, indent=4)
