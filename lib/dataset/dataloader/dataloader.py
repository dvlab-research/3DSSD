class Dataset:
    """
    Template dataset loader and producer
    """
    def __init__(self, mode, split, img_list, is_training, workers_num):
        raise NotImplementedError
    
    def __len__(self):
        raise NotImplementedError

    # load data
    def load_samples(self, sample_idx, pipename):
        raise NotImplementedError

    def load_batch(self, batch_size):
        raise NotImplementedError

    # Preprocess data
    def preprocess_samples(self, indices):
        raise NotImplementedError 

    def generate_mixup_sample(self, sample_dict):
        raise NotImplementedError

    def preprocess_batch(self):
        raise NotImplementedError

    # Evaluation
    def set_evaluation_tensor(self, model):
        raise NotImplementedError

    def evaluate_map(self, sess, feeddict_producer, pred_list, val_size, cls_thresh, log_dir, placeholders=None):
        raise NotImplementedError

    def evaluate_recall(self, sess, feeddict_producer, pred_list, val_size, iou_threshold, log_dir, placeholders=None):
        raise NotImplementedError

    def logger_and_select_best_map(self, result_list, log_string):
        raise NotImplementedError

    def logger_and_select_best_recall(self, result_list, log_string):
        raise NotImplementedError

    # Save predictions
    def save_predictions(self, sess, feeddict_producer, pred_list, val_size, cls_thresh, log_dir, placeholders=None):
        raise NotImplementedError
