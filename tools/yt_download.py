from pytube import YouTube
import pandas as pd
import cv2
from tqdm import tqdm
import os

if __name__ == '__main__':
    music_list = pd.read_csv('./PianoYT/pianoyt.csv', names=["index", "link", "train/test", "crop_minY", "crop_maxY", "crop_minX", "crop_maxX"])
    
    # yt.streams로 사용 가능한 youtube 객체(동영상, 음성 등) 확인 가능
    # filter로 원하는 형태의 stream 가져오기 가능
    # video들은 25fps

    os.makedirs("./videos/train/", exist_ok=True)
    os.makedirs("./videos/test/", exist_ok=True)
    for _, (is_train, music_link, music_idx) in tqdm(music_list[['train/test', 'link', 'index']].iterrows(), total=music_list.shape[0]):
        if is_train == 1:
            output_path = "./videos/train/"
        else:
            output_path = "./videos/test/"
        try:
            yt = YouTube(music_link)
            yt.streams.filter(mime_type="video/mp4", res="1080p").first().download(output_path=output_path, filename_prefix=f"{music_idx}_")
        except Exception:
            print(f"failed to download : {music_idx}, {music_link}")