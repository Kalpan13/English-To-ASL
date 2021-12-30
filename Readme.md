Setup Guide

1. Install the requirements using `pip install -r requirements.txt`

2. Run server.py using `python server.py` command. The server will take sometime to load during the first call because tensorflow library takes sometime to load its checkpoints

3. Use Record Audio button to speak. Try to speak slowly and clearly. Speech Recognition API fails to recognise words in some scenarios. 

4. The video containing ASL gestures will be rendered after the wait of a few seconds.



Code details 

download_video.py : Used for downloading ASL videos from website
english_to_asl.py : Used for converting English Language to ASL
server.py : Used for rendering HTML pages and videos
speech_to_text.py : Used for converting Spoken sentence to text
ENG_ASL_Attention_Mechanism.ipynb : File used for model tranining. (I could not attach the Data file because of the size)
static : stores the ASL sign videos generated
ASL_gestures : stores gestures for each words (scrapped from website)



Note : I had to remove tensorflow checkpoints of model because blackboard was not accepting larger zip files
