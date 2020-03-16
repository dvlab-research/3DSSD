import os, sys
import tensorflow as tf
import numpy as np
import argparse
import pprint
import importlib
import datetime

from tensorflow.python import pywrap_tensorflow
from tensorflow.python.client import device_lib as _device_lib

from core.config import cfg, cfg_from_file, cfg_from_list, assert_and_infer_cfg
from core.trainer_utils import *
from dataset.dataloader import choose_dataset
from dataset.feeddict_builder import FeedDictCreater
from modeling import choose_model


def parse_args():
    parser = argparse.ArgumentParser(description='Trainer')
    parser.add_argument('--cfg', required=True, help='Config file for training')
    parser.add_argument('--img_list', default='train', required=True, help='Train/Val/Trainval list')
    parser.add_argument('--split', default='training', help='Dataset split')
    parser.add_argument('--restore_model_path', default=None, help='Restore model path e.g. log/model.ckpt [default: None]')

    return args


class trainer:
    def __init__(self, args):
        self.batch_size = cfg.TRAIN.CONFIG.BATCH_SIZE
        self.gpu_num = cfg.TRAIN.CONFIG.GPU_NUM
        self.num_workers = cfg.DATA_LOADER.NUM_THREADS 
        self.log_dir = cfg.MODEL.PATH.CHECKPOINT_DIR
        self.max_iteration = cfg.TRAIN.CONFIG.MAX_ITERATIONS
        self.checkpoint_interval = cfg.TRAIN.CONFIG.CHECKPOINT_INTERVAL
        self.summary_interval = cfg.TRAIN.CONFIG.SUMMARY_INTERVAL
        self.restore_model_path = args.restore_model_path
        self.is_training = True

        # gpu_num
        self.gpu_num = min(self.gpu_num, len(self._get_available_gpu_num()))

        # save dir
        datetime_str = str(datetime.datetime.now())
        self.log_dir = os.path.join(self.log_dir, datetime_str)
        if not os.path.exists(self.log_dir): os.makedirs(self.log_dir)
        self.log_file = open(os.path.join(self.log_dir, 'log_train.txt'), 'w')
        self.log_file.write(str(args)+'\n')
        self._log_string('**** Saving models to the path %s ****'%self.log_dir)
        self._log_string('**** Saving configure file in %s ****'%self.log_dir)
        os.system('cp \"%s\" \"%s\"' % (args.cfg, self.log_dir))

        # dataset
        dataset_func = choose_dataset()
        self.dataset = dataset_func('loading', split=args.split, img_list=args.img_list, is_training=self.is_training, workers_num=self.num_workers)
        self.dataset_iter = self.dataset.load_batch(self.batch_size * self.gpu_num)
        self._log_string('**** Dataset length is %d ****'%len(self.dataset))

        # solver
        self.global_step = tf.contrib.framework.get_or_create_global_step()
        self.bn_decay = get_bn_decay(self.global_step)
        self.learning_rate = get_learning_rate(self.global_step)

        # models
        self.model_func = choose_model() 
        self.model_list, self.tower_grads, self.total_loss_gpu, self.losses_list, self.params, self.extra_update_ops = self._build_model_list()
        tf.summary.scalar('total_loss', self.total_loss_gpu)

        # feeddict
        self.feeddict_producer = FeedDictCreater(self.dataset_iter, self.model_list, self.batch_size)

        # optimizer
        if cfg.SOLVER.TYPE == 'SGD':
            self.optimizer = tf.train.MomentumOptimizer(self.learning_rate, momentum=cfg.SOLVER.MOMENTUM)
        elif cfg.SOLVER.TYPE == 'Adam':
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        with tf.device('/gpu:0'):
            self.grads = average_gradients(self.tower_grads)
            self.update_op = [self.optimizer.apply_gradients(zip(self.grads, self.params), global_step=self.global_step)]
        self.update_op.extend(self.extra_update_ops)
        self.train_op = tf.group(self.update_op)
        
        # tensorflow training ops 
        self.sess = tf.Session()
        self.saver = tf.train.Saver()
        self.merged = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter(os.path.join(self.log_dir, 'train'), sess.graph)
   
        # initialize model
        self._initialize_model()


    def _get_available_gpu_num(self): 
        local_device_protos = _device_lib.list_local_devices()
        return [x.name for x in local_device_protos if x.device_type == 'GPU']


    def _log_string(self, out_str):
        self.log_file.write(out_str+'\n')
        self.log_file.flush()
        print(out_str) 

    def _build_model_list(self):
        # build model, losses and tower gradient
        model_list = []
        tower_grads = []
        losses_dict = dict()
        total_loss_gpu = []
        for i in range(self.gpu_num):
            with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
                with tf.device('/gpu:%d'%(i)), tf.name_scope('gpu_%d'%(i)) as scope:
                    model = self.model_func(self.batch_size, self.is_training) 
                    model.model_forward(self.bn_decay)
                    model_list.append(model)

                    losses = tf.get_collection('losses', scope)
                    total_loss = tf.add_n(losses, name='total_loss')
                    for l in losses:
                        l_name = '/'.join(l.name.split('/')[1:])
                        l_name = l_name.split('_')[:-1]
                        l_name = '_'.join(l_name)
                        if l_name not in losses_dict.keys():
                            losses_dict[l_name] = []
                        losses_dict[l_name].append(l)
                    params = tf.trainable_variables()
                    grads = tf.gradients(total_loss, params)
                    clipped_gradients, gradient_norm = tf.clip_by_global_norm(grads, 5.0)
                    tower_grads.append(clipped_gradients) 
                    total_loss_gpu.append(total_loss)
         
                    if i == 0: # update bn
                        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        losses_list = []
        for k, v in losses_dict.items():
            losses_list.append(tf.identity(tf.reduce_mean(v), k))
        total_loss_gpu = tf.reduce_mean(total_loss_gpu)
        return model_list, tower_grads, total_loss_gpu, losses_list, params, extra_update_ops

    def _initialize_model(self):
        init = tf.global_variables_initializer()
        self.sess.run(init) 

        if self.restore_model_path is not None:
            # restore from a pre-trained file 
            path = os.path.expanduser(os.path.expandvars(args.restore_model_path))
            var_dict = get_variables_in_checkpoint_file(path)
            trainable_variables = tf.trainable_variables()
            global_variables = tf.global_variables()
            variables_to_restore = {}
            for var in global_variables:
                if 'global_step' in var.name: continue
                var_name = var.name.split(':')[0]
                if var_name in var_dict:
                    variables_to_restore[var.name.split(':')[0]] = var
            self._log_string('transferring from ' + path)
            restorer = tf.train.Saver(variables_to_restore)
            restorer.restore(self.sess, path)
        

    def train(self):
        for step in range(self.max_iteration):
            if step % self.checkpoint_interval == 0: # save model
                global_step_np = tf.train.global_step(self.sess, self.global_step) 

                self.saver.save(self.sess,
                                save_path=os.path.join(self.log_dir, 'model'),
                                global_step=global_step_np)
                self._log_string('Step {} / {}, Checkpoint saved to {}-{:08d}'.format(
                                 step, self.max_iteration,
                                 os.path.join(self.log_dir, 'model'), global_step_np))

            feed_dict = self.feeddict_producer.create_feed_dict()
 
            if step % self.summary_interval == 0:
                cur_time = time.time()
                time_elapsed = cur_time - last_time
                last_time = cur_time

                _, train_op_loss, summary_out, *losses_list_np = sess.run([self.train_op, self.total_loss_gpu, self.merged] + self.losses_list, feed_dict=feed_dict)

                self._log_string('**** STEP %08d ****'%step)
                self._log_string('Step {}, Total Loss {:0.3f}, Time Elapsed {:0.3f} s'.format(
                    step, train_op_loss, time_elapsed))
                for loss, loss_name in zip(losses_list_np, self.losses_list):
                    self._log_string('Loss: {}: {:0.3f}'.format(loss_name.name, loss))
                self.train_writer.add_summary(summary_out, step)
            else:
                sess.run(self.train_op, feed_dict=feed_dict)
        

if __name__ == '__main__':
    args = parse_args()
    cfg_from_file(args.cfg)

    cur_trainer = trainer(args) 
    cur_trainer.train() 
    print("**** Finish training steps ****")
