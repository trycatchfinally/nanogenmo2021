# nanogenmo2021
## Concept
An AI watches and describes the individual frames of the music videos of the songs you remember in autobiographical order.

## General flow
1. Save individual frames from a video (see generate_frames.py or use `ffmpeg -i $vid.mp4 -r 1 -f image2 $vid-%4d.png`)
2. Use Azure Computer Vision to `analyze()` the image file (throttled at 20 images per 61 seconds for the free tier)
3. Expand the analysis result into a list of phrases (`extract_text()`)
4. Remove redundant or overlapping phrases within a single line (one image, one paragraph)
5. Remove repeated phrases across multiple lines with a configurable window (`gen_one_chapter()`)
6. Possibly randomize the phrase ordering within a line to make it more interesting (`textmods.py` and `test_reduction.py`)
7. Save each chapter to a .md file and one big .md file
8. `run_pandoc` to generate a PDF

## Requirements/configuration
Sign up for a free tier Microsoft Azure account with Computer Vision enabled:
> https://azure.microsoft.com/en-us/services/cognitive-services/computer-vision/#getting-started

Add a local-only .gitignore'd file called `secrets.py` containing
```
VISION_KEY = "some hex string from Azure"
```

Python packages needed (at least):
```
pip install opencv-python azure-cognitiveservices-vision-computervision pandas ratelimiter tqdm loguru jsonpickle pytest
```

To get PDF output, you'll need a LaTeX distribution and pandoc: https://miktex.org/ for Windows and https://pandoc.org/ for all platforms.
(The markdown generated contains some \pagebreak's.)

You may or may not want `ffmpeg` to generate the frames. (It will have different results than `generate_frames.py` which I included to make everything mostly self-contained.)

## Running
`pytest test_example.py` shows the whole workflow for the public domain `examples/Archive1951.mp4` file, downloaded from https://archive.org/details/Survival1951 .

`computer_vision.py` is the main program which takes `book.xlsx` and the videos + frames (generated offline) in the 'easy/' directory.

## Generated directories
For debugging/inspection purposes, these directories are created:

| Directory | Contents |
|-----------|----------|
| .cache | mostly binary pickle files of Azure responses to avoid re-querying for the same image |
| .debug | json files of the Azure responses |
| .debug-class | see what Azure classifies as Adult, Gory, and Racy! |
| .debug-lines | the un-simplified, un-randomized output  |
| .debug-parts | the individual 'chapters' in .md |
