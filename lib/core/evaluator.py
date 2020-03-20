import os, sys
import tensorflow as tf
import numpy as np
import argparse
import pprint
import importlib
import time

from tensorflow.python import pywrap_tensorflow
from tensorflow.python.client import device_lib as _device_lib

from core.config import cfg, cfg_from_file, cfg_from_list, assert_and_infer_cfg
from dataset.dataloader import choose_dataset
from dataset.feeddict_builder import FeedDictCreater
from modeling import choose_model
import dataset.maps_dict as maps_dict

def parse_args():
    parser = argparse.ArgumentParser(description='Trainer')
    parser.add_argument('--cfg', required=True, help='Config file for training')
    parser.add_argument('--restore_model_path', required=True, help='Restore model path e.g. log/model.ckpt [default: None]')
    parser.add_argument('--img_list', default='val', help='Train/Val/Trainval list')
    parser.add_argument('--split', default='training', help='Dataset split')

    # some evaluation threshold
    parser.add_argument('--cls_threshold', default=0.3, help='Filtering Predictions')
    parser.add_argument('--eval_interval_secs', default=300, help='Sleep after one evaluation loop')
    parser.add_argument('--no_gt', action='store_true', help='Used for test set')
    args = parser.parse_args()

    return args

class evaluator:
    def __init__(self, args):
        self.batch_size = cfg.TRAIN.CONFIG.BATCH_SIZE
        self.gpu_num = cfg.TRAIN.CONFIG.GPU_NUM
        self.num_workers = cfg.DATA_LOADER.NUM_THREADS
        self.log_dir = cfg.MODEL.PATH.EVALUATION_DIR
        self.is_training = False

        self.cls_thresh = float(args.cls_threshold)
        self.eval_interval_secs = args.eval_interval_secs
        self.restore_model_path = args.restore_model_path

        # save dir
        self.log_dir = os.path.join(self.log_dir, self.restore_model_path)
        if not os.path.exists(self.log_dir): os.makedirs(self.log_dir)
        self.log_file = open(os.path.join(self.log_dir, 'log_train.txt'), 'w')
        self.log_file.write(str(args)+'\n')
        self._log_string('**** Saving Evaluation results to the path %s ****'%self.log_dir)

        # dataset
        dataset_func = choose_dataset()
        self.dataset = dataset_func('loading', split=args.split, img_list=args.img_list, is_training=self.is_training, workers_num=self.num_workers)
        self.dataset_iter = self.dataset.load_batch(self.batch_size * self.gpu_num)
        self._log_string('**** Dataset length is %d ****'%len(self.dataset))
        self.val_size = len(self.dataset)

        # model list
        self.model_func = choose_model()
        self.model_list, self.pred_list = self._build_model_list()

        # feeddict
        self.feeddict_producer = FeedDictCreater(self.dataset_iter, self.model_list, self.batch_size)

        # evaluation tools 
        self.last_eval_model_path = None
        self.last_best_model = None
        self.last_best_result = -1 
        self.saver = tf.train.Saver()


    def _build_model_list(self):
        model_list = []
        model = self.model_func(self.batch_size, self.is_training)
        model.model_forward()
        model_list.append(model)

        # get prediction results, bs = 1
        pred_list = self.dataset.set_evaluation_tensor(model)
        
        return model_list, pred_list 


    def _log_string(self, out_str):
        self.log_file.write(out_str+'\n')
        self.log_file.flush()
        print(out_str)

   
    def evaluate(self):
        while True:
            start = time.time()
            self._log_string('Starting Evaluation at ' + time.strftime('%Y-%m-%d-%H:%M:%S', time.gmtime()))
            checkpoint_dirs = self.restore_model_path
            
            if os.path.isdir(checkpoint_dirs):
                cur_model_path = tf.train.latest_checkpoint(checkpoint_dirs)
            else:
                cur_model_path = checkpoint_dirs
            if not cur_model_path:
                self._log_string('No model found in %s. Will try again in %d seconds' %(checkpoint_dirs, self.eval_interval_secs))
            elif cur_model_path == self.last_eval_model_path:
                self._log_string('Found already evaluated checkpoint. Will try again in %d seconds'%self.eval_interval_secs)
            else: # evaluating
                self.last_eval_model_path = cur_model_path

                with tf.Session() as sess:
                    self._log_string('**** Evluation New Result ****')
                    self._log_string('Assign From checkpoint: %s'%self.last_eval_model_path)

                    self.saver.restore(sess, self.last_eval_model_path)
                    result_list = self.dataset.evaluation(sess, self.feeddict_producer, self.pred_list, self.val_size, self.cls_thresh, self.log_dir) 
                    cur_result = self.dataset.logger_and_select_best(result_list, self._log_string)

                    if cur_result > self.last_best_result:
                        # if cur_result is larger, save the current weight model
                        if self.last_best_model is not None:
                            # if last model is not none, then we remove that
                            last_best_model_name = os.path.basename(self.last_best_model)
                            last_best_model_path = os.path.join(self.log_dir, last_best_model_name)
                            os.system('rm \"%s\".*'%last_best_model_path)
                        os.system('cp \"%s\".* \"%s\"' % (cur_model_path, self.log_dir))
                        self.last_best_model = cur_model_path
                        self.last_best_result = cur_result

                    self._log_string(self.last_eval_model_path)
                    self._log_string("**** Done !!! ****")

            time_to_next_eval = start + self.eval_interval_secs - time.time()
            if time_to_next_eval > 0:
                time.sleep(time_to_next_eval)
 
        


if __name__ == '__main__':
    args = parse_args()
    cfg_from_file(args.cfg)

    # set bs, gpu_num and workers_num to be 1
    cfg.TRAIN.CONFIG.BATCH_SIZE = 1 # only support bs=1 when testing
    cfg.TRAIN.CONFIG.GPU_NUM = 1
    cfg.DATA_LOADER.NUM_THREADS = 1
    if args.no_gt:
        cfg.TEST.WITH_GT = False

    cur_evaluator = evaluator(args)
    cur_evaluator.evaluate()
    print("**** Finish evaluation steps ****")
