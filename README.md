# Known-Contact Recognition Assistant

Detects people, cats, and dogs in a video with YOLOv8-nano, estimates an
approximate distance for each, and shows an information panel for enrolled
contacts using dlib-based face recognition. Cats and dogs are excluded from
face recognition.

This package contains the same code in two forms: a Python script
(`known_contact_assistant.py`) and a notebook (`known_contact_assistant.ipynb`).
Use whichever suits you; see "Run" below.

## Setup

**Dependencies:**
```
pip install ultralytics opencv-python numpy
pip install cmake dlib face_recognition
```
`dlib` may need a C++ compiler and CMake; see the `dlib` install docs if the
build fails. (In Google Colab, `dlib` is usually already available, so the
notebook's install cell is normally enough.)

**Models (pretrained, no fine-tuning):**
- [YOLOv8-nano](https://github.com/ultralytics/ultralytics) (Ultralytics),
  COCO weights. Classes used: `0 person`, `15 cat`, `16 dog`. The weights file
  `yolov8n.pt` is downloaded automatically on first run.
- [`face_recognition`](https://github.com/ageitgey/face_recognition)
  (Adam Geitgey), a Python library that wraps dlib for face detection,
  embedding, and matching.
- [dlib](https://github.com/davisking/dlib) (Davis King), the C++ machine
  learning library whose HOG-based face detector and ResNet-based
  face-embedding model are used under the hood.

## Input files

Put these in the same folder as the script or notebook (or in the Colab
working directory):

| File | Purpose |
|---|---|
| `test_video.mp4` | Input video clip |
| `person-1.PNG` | Reference face image for contact `XYZ` |
| `person-2.PNG` | Reference face image for contact `ABC` |

To enroll a contact, add an entry to `KNOWN_CONTACTS` with a clear,
front-facing photo (and only enroll people who have agreed to it):
```python
KNOWN_CONTACTS = {
    "ABC": {"image_file": "person-2.PNG", "note": "Undergrad Thesis Mentee"},
}
```
File names are case-sensitive on Linux and Colab, so `person-2.PNG` and
`person-2.png` are different files.

## Run

**Option A: Python script**

1. Install the dependencies (above).
2. Place the input files next to `known_contact_assistant.py`.
3. If your video has a different name, change `VIDEO_PATH` near the top of the
   script.
4. Run:
   ```
   python known_contact_assistant.py
   ```
5. The annotated video is saved as `known_contacts_demo.mp4` (set by
   `OUT_PATH`), and the console prints the number of frames and the time taken.

**Option B: Notebook**

1. Open `known_contact_assistant.ipynb` in Jupyter or Google Colab.
2. In Colab, upload the three input files to the working folder using the file
   sidebar.
3. Run the cells from top to bottom. The first code cell installs the
   dependencies with `!pip install` (this is why those lines are not in the
   `.py` file), and the last cell, `main()`, runs the pipeline.
4. The annotated video is saved as `known_contacts_demo.mp4`. In Colab, download
   it from the file sidebar when it finishes.

## Settings

These are in the configuration section at the top of the script (or in the
configuration cell of the notebook):

| Setting | Meaning |
|---|---|
| `VIDEO_PATH`, `OUT_PATH` | Input clip and output video names |
| `CONF` | Minimum detection confidence |
| `SOCIAL_TRIGGER_DIST_M` | Face recognition runs only for people closer than this estimated distance |
| `FACE_CHECK_EVERY_N_FRAMES` | How often face recognition is re-run; the last result is reused in between |
| `FACE_MATCH_TOLERANCE` | Match strictness (lower is stricter) |
| `PANEL_W`, `PANEL_H` | Size of the contact panel in pixels |
