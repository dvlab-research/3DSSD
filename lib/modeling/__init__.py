from .single_stage_detector import SingleStageDetector
from .double_stage_detector import DoubleStageDetector
from core.config import cfg

def choose_model():
    model_dict = {
        'SingleStage': SingleStageDetector,
        'DoubleStage': DoubleStageDetector, 
    } 

    return model_dict[cfg.MODEL.TYPE]
