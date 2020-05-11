from fvcore.common.config import CfgNode


_C = CfgNode()

_C.MODE = 'train'

_C.TRAIN = CfgNode()
_C.TRAIN.NUM_EPOCHS = 10
_C.TRAIN.BATCH_SIZE = 4
_C.TRAIN.ATTENTION_STEPS = 2
_C.TRAIN.LR = 0.001
_C.TRAIN.CHECKPOINT_FILE_PATH = ''
_C.TRAIN.VIDEO_ANNOTATION_FILE = '../datasets/activitynet/captions/train.json'
_C.TRAIN.FEATURE_LENGTHS_PATH = '../datasets/activitynet/captions/train_feature_lengths.json'

_C.VAL = CfgNode()
_C.VAL.VIDEO_ANNOTATION_FILE = '../datasets/activitynet/captions/val_1.json'
_C.VAL.FEATURE_LENGTHS_PATH = '../datasets/activitynet/captions/val_feature_lengths.json'

_C.DATA = CfgNode()
_C.DATA.ENV_FEATURE_DIR = '../datasets/activitynet/env_features/'
_C.DATA.AGENT_FEATURE_DIR = '../datasets/activitynet/agent_features/'
_C.DATA.RESULT_PATH = './results/results.json'
_C.DATA.FIGURE_PATH = './results/result_figure.jpg'
_C.DATA.SCORE_PATH = './results/scores.json'
_C.DATA.TEMPORAL_DIM = 100
_C.DATA.FEATURE_DIM = 2304
_C.DATA.TARGET_FPS = 30
_C.DATA.SAMPLING_RATE = 16

_C.MODEL = CfgNode()
_C.MODEL.CHECKPOINT_DIR = 'checkpoints/'
_C.MODEL.CHECKPOINT_BEST_RECORDS = 'checkpoints/best_scores.json'

_C.BMN = CfgNode()
_C.BMN.NUM_SAMPLES = 32
_C.BMN.NUM_SAMPLES_PER_BIN = 3
_C.BMN.POST_PROCESS = CfgNode()
_C.BMN.POST_PROCESS.SOFT_NMS_ALPHA = 0.4
_C.BMN.POST_PROCESS.SOFT_NMS_LOW_THRESHOLD = 0.5
_C.BMN.POST_PROCESS.SOFT_NMS_HIGH_THRESHOLD = 0.9
_C.BMN.PROP_BOUNDARY_RATIO = 0.5
_C.BMN.POST_PROCESS.NUM_THREADS = 8
_C.BMN.POST_PROCESS.RESULTS_FILE = './outputs/post_processing_results.json'

_C.NUM_GPUS = 1


def _assert_and_infer_cfg(cfg):
    assert cfg.TRAIN.BATCH_SIZE % cfg.NUM_GPUS == 0
    return cfg


def get_cfg():
    """
    Get a copy of the default config.
    """
    return _assert_and_infer_cfg(_C.clone())
