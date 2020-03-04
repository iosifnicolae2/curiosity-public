### Steps
1. Download the repository as an archive from https://github.com/iosifnicolae2/curiosity
2. Start a Jupyter Notebook
3. Upload the file to `/storage` on the Jupyter Notebook
4. Open a terminal, enter `bash` and then install `unzip`: `apt install unzip`
5. Unzip the archive `unzip curiosity-master.zip && cd curiosity-master`
6. Install Python dependencies: `pip install -e .`
7. Start the training: `python app/sm_2d/train.py`