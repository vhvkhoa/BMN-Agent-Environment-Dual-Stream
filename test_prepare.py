from os import path
import glob
import cv2
import json
from collections import defaultdict
from tqdm import tqdm

DATA_PATH = "test_rescaled"
SAVE_PATH = "test_cropped_rescaled"

""" Actions
Ambiguous = 0
BaseballPitch = 7
BasketballDunk = 9
Billiards = 12
CleanAndJerk = 21
CliffDiving = 22
CricketBowling = 23
CricketShot = 24
Diving = 26
FrisbeeCatch = 31
GolfSwing = 33
HammerThrow = 36
HighJump = 40
JavelinThrow = 45
LongJump = 51
PoleVault = 68
Shotput = 79
SoccerPenalty = 85
TennisSwing = 92
ThrowDiscus = 93
VolleyballSpiking = 97
"""

WFEATURE = 16
WSCALE = 128
WSTRIDE = 128

f_feature = WFEATURE
f_stride = WSTRIDE * WFEATURE
f_scale = WSCALE * WFEATURE

dataset = defaultdict(dict)
for name in tqdm(sorted(glob.glob(DATA_PATH + "/*"))):
    cap = cv2.VideoCapture(name)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    assert length % f_scale == 0, 'Video length %d is not divided by %d.' % (length, f_scale)
    fps = cap.get(cv2.CAP_PROP_FPS)
    assert fps == 30, 'Video fps should be 30. %d found.' % fps

    for i in range(0, length, f_stride):
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(SAVE_PATH + "/{}-{}.mp4".format(name, i), fourcc, fps, size)
        for _ in range(f_scale):
            ret, frame = cap.read()
            writer.write(frame)
            assert ret is True, 'Reading video failed.'
        writer.release()
