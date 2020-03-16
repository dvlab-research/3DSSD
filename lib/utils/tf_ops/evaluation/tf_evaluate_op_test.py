import tensorflow as tf
import numpy as np
from tf_evaluate import evaluate

class GroupPointTest(tf.test.TestCase):
  def test(self):
    with self.test_session() as sess:
      dets = tf.constant([
             [2, 645.60, 167.95, 680.98, 198.93, -1.65, 4.59, 1.32, 45.84, 1.86, 0.60, 2.02, -1.55, 0.80],
             [0, 387.63, 181.54, 423.81, 203.12, 1.85, -16.53, 2.39, 58.49, 1.67, 1.87, 3.69, 1.57, 0.99], 
             [1, 712.40, 143.00, 810.73, 307.92, -0.20, 1.84, 1.47, 8.41, 1.89, 0.48, 1.20, 0.01, 0.70], 
             [0, 614.24, 181.78, 727.31, 284.77, 1.55, 1.00, 1.75, 13.22, 1.57, 1.73, 4.15, 1.62, 0.99],
          ])
      names = tf.constant(['/root/kitti_native_evaluation/gtfiles/000001.txt', '/root/kitti_native_evaluation/gtfiles/000000.txt', '/root/kitti_native_evaluation/gtfiles/000003.txt'])
      numlist = tf.constant([2, 1, 1])
      outs = evaluate(dets, names, numlist) 
      print(outs)
      pi, ai, pg, ag, p3, a3 = sess.run(outs)
      print(pi)
      print(pi.shape)
      print(pg)
      print(pg.shape)
      print(p3)
      print(p3.shape)

  def test_grad(self):
    pass

if __name__=='__main__':
  tf.test.main() 
