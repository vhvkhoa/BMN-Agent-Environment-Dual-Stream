from fvcore.common.config import CfgNode


_C = CfgNode()

_C.TRAIN = CfgNode()
_C.TRAIN.BATCH_SIZE = 1
_C.TRAIN.ATTENTION_BATCH_SIZE = 128
_C.TRAIN.LR = 0.001
_C.TRAIN.CHECKPOINT_FILE_PATH = ""

_C.DATA = CfgNode()
_C.DATA.ENV_FEATURE_DIR = "../datasets/tmp_anet/env_features/"
_C.DATA.AGENT_FEATURE_DIR = "../datasets/tmp_anet/agent_features/"
_C.DATA.VIDEO_ID_FILE = "../datasets/tmp_anet/tmp_ids.json"
_C.DATA.VIDEO_ANNOTATION_FILE = "../datasets/tmp_anet/tmp.json"
_C.DATA.SAMPLE_MASK_DIR = "../datasets/tmp_anet/sample_masks"
_C.DATA.TEMPORAL_DIM = 100
_C.DATA.FEATURE_DIM = 2304
_C.DATA.TARGET_FPS = 30
_C.DATA.SAMPLING_RATE = 16

_C.MODEL = CfgNode()

_C.BMN = CfgNode()
_C.BMN.NUM_SAMPLES = 32
_C.BMN.NUM_SAMPLES_PER_BIN = 3
_C.BMN.SOFT_NMS_ALPHA = 0.4
_C.BMN.SOFT_NMS_LOW_THRESHOLD = 0.5
_C.BMN.SOFT_NMS_HIGH_THRESHOLD = 0.9
_C.BMN.PROP_BOUNDARY_RATIO = 0.5

_C.NUM_GPUS = 1


def _assert_and_infer_cfg(cfg):
    assert cfg.TRAIN.BATCH_SIZE % cfg.NUM_GPUS == 0

    return cfg


def get_cfg():
    """
    Get a copy of the default config.
    """
    return _assert_and_infer_cfg(_C.clone())