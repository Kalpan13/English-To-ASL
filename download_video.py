from bs4 import BeautifulSoup
import requests
import os
from moviepy.editor import *

BASE_URL = "https://www.signingsavvy.com/sign/"
VIDEO_BASE_URL = "https://www.signingsavvy.com/"
ASL_VIDEOS_FOLDER_NAME = "ASL_Gestures"

def get_video_url(word_url : str, word : str):
    """To extract ASL video URL from word URL

    Args:
        word_url (str): URL of the word 
        word (str): word
    """
    # Get HTML page
    response = requests.get(word_url)
    soup = BeautifulSoup(response.content, features="html.parser")

    # Handle case for multiple gestures
    multiple_records = soup.find_all("div", {"class": "search_results"})
    if len(multiple_records) > 0:
        possible_words = multiple_records[0].find_all("li")
        possible_words = possible_words[0]
        if len(possible_words) > 1:
            possible_words = multiple_records[0].find_all("a")
            for possible_word in possible_words:
                if word in possible_word.get('href','').lower():
                    # Take the first matching word
                    get_video_url(BASE_URL+possible_word.get('href',''), word)
                    break
    else:
        # Extract video URL from src attr
        video_tags = soup.find_all("source")
        if len(video_tags) > 0:
            video_url = video_tags[0]["src"]
            download_file(VIDEO_BASE_URL+video_url, word)

def download_file(url : str, filename : str):
    """Download video from URL

    Args:
        url (str): Video URL to be downloaded
        filename (str): Name of the video file
    """
    # Download a video file
    filename = filename + ".mp4"
    r = requests.get(url, stream=True)
    with open(f"./{ASL_VIDEOS_FOLDER_NAME}/"+filename, 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024): 
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)
    
def generate_ASL_gesture(sentence : str):
    """
    Generate ASL gesture video from sentence

    Args:
        sentence (str): ASL text sentence
    """
    if not os.path.isdir(ASL_VIDEOS_FOLDER_NAME):
        os.mkdir(ASL_VIDEOS_FOLDER_NAME)

    for word in sentence.split(" "):
        if not word_present(word):
            print("Downloading word :", word)
            get_video_url(BASE_URL+word, word)

    merge_sign_videos(sentence)

def merge_sign_videos(sentence : str):
    """Merge sign videos

    Args:
        sentence (str): ASL text sentence
    """
    merged_videos = []
    file_name = sentence.replace(" ","_")
    for word in sentence.split(" "):
        if word_present(word):
            video = VideoFileClip(f"./{ASL_VIDEOS_FOLDER_NAME}/"+word + ".mp4")
            merged_videos.append(video)
            final_clip = concatenate_videoclips(merged_videos)
            print("Writing Video")
            final_clip.to_videofile(f"./static/{file_name}.mp4", fps=24, remove_temp=False)

def word_present(word):
    return word+".mp4" in os.listdir(path=f'./{ASL_VIDEOS_FOLDER_NAME}')

if __name__ == '__main__':
    generate_ASL_gesture("hey ball")