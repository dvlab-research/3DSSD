from .single_stage_detector import SingleStageDetector
from core.config import cfg

def choose_model():
    model_dict = {
        'SingleStage': SingleStageDetector,
    } 

    return model_dict[cfg.MODEL.TYPE]
