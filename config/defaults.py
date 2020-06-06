from fvcore.common.config import CfgNode


_C = CfgNode()

_C.GPU_IDS = [0]
_C.MODE = 'testing'
_C.USE_ENV = True
_C.USE_AGENT = True

_C.TRAIN = CfgNode()
_C.TRAIN.NUM_EPOCHS = 10
_C.TRAIN.BATCH_SIZE = 2
_C.TRAIN.STEP_PERIOD = 16
_C.TRAIN.ATTENTION_STEPS = 1
_C.TRAIN.LR = 0.0001
_C.TRAIN.CHECKPOINT_FILE_PATH = ''
_C.TRAIN.VIDEO_ANNOTATION_FILE = '../datasets/activitynet/annotations/train.json'
_C.TRAIN.LOG_DIR = './agent_env_runs/'

_C.VAL = CfgNode()
_C.VAL.VIDEO_ANNOTATION_FILE = '../datasets/activitynet/annotations/val_1.json'

_C.TEST = CfgNode()
_C.TEST.CHECKPOINT_PATH = '../checkpoints/best_auc.pth'

_C.DATA = CfgNode()
_C.DATA.ANNOTATION_FILE = '../datasets/activitynet/annotations/activity_net.v1-3.min.json'
_C.DATA.ENV_FEATURE_DIR = '../datasets/activitynet/new_env_features/'
_C.DATA.AGENT_FEATURE_DIR = '../datasets/activitynet/new_agent_features/'
_C.DATA.RESULT_PATH = './results/results.json'
_C.DATA.FIGURE_PATH = './results/result_figure.jpg'
_C.DATA.TEMPORAL_DIM = 100
_C.DATA.FEATURE_DIM = 2304
_C.DATA.TARGET_FPS = 30
_C.DATA.SAMPLING_RATE = 16

_C.MODEL = CfgNode()
_C.MODEL.SCORE_PATH = './checkpoints/scores.json'
_C.MODEL.CHECKPOINT_DIR = 'checkpoints/'
_C.MODEL.ATTENTION_HEADS = 4
_C.MODEL.ATTENTION_LAYERS = 1
_C.MODEL.HIDDEN_DIM_1D = 1024  # 256
_C.MODEL.HIDDEN_DIM_2D = 512   # 128
_C.MODEL.HIDDEN_DIM_3D = 1024  # 512

_C.BMN = CfgNode()
_C.BMN.NUM_SAMPLES = 32
_C.BMN.NUM_SAMPLES_PER_BIN = 3
_C.BMN.POST_PROCESS = CfgNode()
_C.BMN.POST_PROCESS.SOFT_NMS_ALPHA = 0.4
_C.BMN.POST_PROCESS.SOFT_NMS_LOW_THRESHOLD = 0.5
_C.BMN.POST_PROCESS.SOFT_NMS_HIGH_THRESHOLD = 0.9
_C.BMN.PROP_BOUNDARY_RATIO = 0.5
_C.BMN.POST_PROCESS.NUM_THREADS = 8

_C.NUM_GPUS = 1


def _assert_and_infer_cfg(cfg):
    assert cfg.TRAIN.BATCH_SIZE % cfg.NUM_GPUS == 0
    return cfg


def get_cfg():
    """
    Get a copy of the default config.
    """
    return _assert_and_infer_cfg(_C.clone())
