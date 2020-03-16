# Modified from tensorpack: https://github.com/ppwwyyxx/tensorpack
import numpy as np
import threading
import multiprocessing as mp
import weakref
from contextlib import contextmanager
from .serialize import loads, dumps
import errno
import uuid
import os
import zmq
import atexit
from itertools import cycle
from copy import copy
from setproctitle import setproctitle
import six
from abc import ABCMeta, abstractmethod
from itertools import chain
import queue

from .logger import *
from .utils import get_rng, set_np_seed

def del_weakref(x):
    o = x()
    if o is not None:
        o.__del__()

@contextmanager
def _zmq_catch_error(name):
    try:
        yield
    except zmq.ContextTerminated:
        print_red("[{}] Context terminated.".format(name))
        raise Exception
    except zmq.ZMQError as e:
        if e.errno == errno.ENOTSOCK:       # socket closed
            print_red("[{}] Socket closed.".format(name))
            raise Exception
        else:
            raise
    except Exception:
        raise

class DataFlowReentrantGuard(object):
    """
    A tool to enforce non-reentrancy.
    Mostly used on DataFlow whose :meth:`get_data` is stateful,
    so that multiple instances of the iterator cannot co-exist.
    """
    def __init__(self):
        self._lock = threading.Lock()

    def __enter__(self):
        self._succ = self._lock.acquire(False)
        if not self._succ:
            raise threading.ThreadError("This DataFlow is not reentrant!")

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._lock.release()
        return False

@six.add_metaclass(ABCMeta)
class DataFlow(object):
    """ Base class for all DataFlow """

    @abstractmethod
    def get_data(self):
        """
        The method to generate datapoints.

        Yields:
            list: The datapoint, i.e. list of components.
        """

    def size(self):
        """
        Returns:
            int: size of this data flow.

        Raises:
            :class:`NotImplementedError` if this DataFlow doesn't have a size.
        """
        raise NotImplementedError()

    def reset_state(self):
        """
        Reset state of the dataflow. It has to be called before producing datapoints.

        For example, RNG **has to** be reset if used in the DataFlow,
        otherwise it won't work well with prefetching, because different
        processes will have the same RNG state.
        """
        pass

class ProxyDataFlow(DataFlow):
    """ Base class for DataFlow that proxies another.
        Every method is proxied to ``self.ds`` unless override by subclass.
    """

    def __init__(self, ds):
        """
        Args:
            ds (DataFlow): DataFlow to proxy.
        """
        self.ds = ds

    def reset_state(self):
        self.ds.reset_state()

    def size(self):
        return self.ds.size()

    def get_data(self):
        return self.ds.get_data()

class DataFromLoader(ProxyDataFlow):
    def __init__(self, data_loader, is_train=True):
        self._data_loader = data_loader
        self._is_train = is_train
        if not self._is_train:
            assert data_loader.batch_size == 1, 'DataLoader should has batch size of one in eval mode.'

    def reset_state(self):
        self._data_iter = iter(self._data_loader)

    def get_data(self):
        while True:
            try:
                example = next(self._data_iter)
            except StopIteration:
                print("end epoch")
                if self._is_train:
                    self._data_iter = iter(self._data_loader)
                    example = next(self._data_iter)
                else:
                    return
            yield example

class DataFromList(ProxyDataFlow):
    def __init__(self, datalist, is_train=True, shuffle=True, batch_size=1):
        self.rng = get_rng()
        self._datalist = datalist
        self._shuffle = shuffle
        self._is_train = is_train
        self._batch_size = batch_size # only works in training

    def get_data(self):
        if self._is_train:
            while True:
                idxses = []
                for b in range(self._batch_size):
                    idxs = np.arange(len(self._datalist))
                    if self._shuffle:
                        self.rng.shuffle(idxs)
                    idxses.append(idxs)
                for i in range(len(idxses[0])):
                    if self._batch_size == 1:
                        yield self._datalist[ idxses[0][i] ]
                    else:
                        yield [ self._datalist[ idxses[b][i] ] for b in range(self._batch_size) ]
        else:
            idxs = np.arange(len(self._datalist))
            if self._shuffle:
                self.rng.shuffle(idxs)
            while True:
                for i in idxs:
                    yield self._datalist[i]

    def reset_state(self):
        self.rng = get_rng()

class _ParallelMapData(ProxyDataFlow):
    def __init__(self, ds, buffer_size):
        assert buffer_size > 0, buffer_size
        self._buffer_size = buffer_size
        self._buffer_occupancy = 0  # actual #elements in buffer

        self.ds = ds

    def _recv(self):
        pass

    def _send(self, dp):
        pass

    def _recv_filter_none(self):
        ret = self._recv()
        assert ret is not None, \
            "[{}] Map function cannot return None when strict mode is used.".format(type(self).__name__)
        return ret

    def _fill_buffer(self, cnt=None):
        if cnt is None:
            cnt = self._buffer_size - self._buffer_occupancy
        try:
            for _ in range(cnt):
                dp = next(self._iter)
                self._send(dp)
        except StopIteration:
            print_yellow(
                "[{}] buffer_size cannot be larger than the size of the DataFlow!".format(type(self).__name__))
            raise
        self._buffer_occupancy += cnt

    def get_data_non_strict(self):
        for dp in self._iter:
            self._send(dp)
            ret = self._recv()
            if ret is not None:
                yield ret

        self._iter = self.ds.get_data()   # refresh
        for _ in range(self._buffer_size):
            self._send(next(self._iter))
            ret = self._recv()
            if ret is not None:
                yield ret

    def get_data_strict(self):
        self._fill_buffer()
        for dp in self._iter:
            self._send(dp)
            yield self._recv_filter_none()
        self._iter = self.ds.get_data()   # refresh

        # first clear the buffer, then fill
        for k in range(self._buffer_size):
            dp = self._recv_filter_none()
            self._buffer_occupancy -= 1
            if k == self._buffer_size - 1:
                self._fill_buffer()
            yield dp

class MapData(ProxyDataFlow):
    """
    Apply a mapper/filter on the DataFlow.

    Note:
        1. Please make sure func doesn't modify the components
           unless you're certain it's safe.
        2. If you discard some datapoints, ``ds.size()`` will be incorrect.
    """

    def __init__(self, ds, func, *args, **kwargs):
        """
        Args:
            ds (DataFlow): input DataFlow
            func (datapoint -> datapoint | None): takes a datapoint and returns a new
                datapoint. Return None to discard this datapoint.
        """
        self.ds = ds
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def get_data(self):
        for dp in self.ds.get_data():
            ret = self.func(copy(dp), *self.args, **self.kwargs)  # shallow copy the list
            if ret is not None:
                yield ret
            else:
                print_yellow('Got empty data [{}]'.format(self.__class__.__name__))

class MultiProcessMapDataZMQ(_ParallelMapData):
    """
    Same as :class:`MapData`, but start processes to run the mapping function,
    and communicate with ZeroMQ pipe.

    Note:
        1. Processes run in parallel and can take different time to run the
           mapping function. Therefore the order of datapoints won't be
           preserved, and datapoints from one pass of `df.get_data()` might get
           mixed with datapoints from the next pass.

           You can use **strict mode**, where `MultiProcessMapData.get_data()`
           is guranteed to produce the exact set which `df.get_data()`
           produces. Although the order of data still isn't preserved.
    """
    class _Worker(mp.Process):
        def __init__(self, identity, map_func, pipename, hwm):
            super(MultiProcessMapDataZMQ._Worker, self).__init__()
            self.identity = identity
            self.map_func = map_func
            self.pipename = pipename
            self.hwm = hwm

        def run(self):
            set_np_seed()
            print(blue('Start data provider'), '{}-{}'.format(self.pipename, self.identity))
            setproctitle('data provider {}-{}'.format(self.pipename, self.identity))
            ctx = zmq.Context()
            socket = ctx.socket(zmq.DEALER)
            socket.setsockopt(zmq.IDENTITY, self.identity)
            socket.set_hwm(self.hwm)
            socket.connect(self.pipename)

            while True:
                dp = loads(socket.recv(copy=False).bytes)
                dp = self.map_func(dp, self.identity)
                # if dp is not None:
                socket.send(dumps(dp), copy=False)
                # get_data_non_strict has filter None-type.
                # else:
                #     print('Got empty data [{}:{}:{}]'.format(self.__class__.__name__, self.pipename, self.identity))

    def __init__(self, ds, nr_proc, map_func, buffer_size=200, strict=False):
        """
        Args:
            ds (DataFlow): the dataflow to map
            nr_proc(int): number of threads to use
            map_func (callable): datapoint -> datapoint | None
            buffer_size (int): number of datapoints in the buffer
            strict (bool): use "strict mode", see notes above.
        """
        _ParallelMapData.__init__(self, ds, buffer_size)
        self.nr_proc = nr_proc
        self.map_func = map_func
        self._strict = strict
        self._procs = []
        self._guard = DataFlowReentrantGuard()

        self._reset_done = False
        self._procs = []

    def _reset_once(self):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.ROUTER)
        self.socket.set_hwm(self._buffer_size * 2)
        pipename = "ipc://@{}-pipe-{}".format('dataflow-map', str(uuid.uuid1())[:8])

        try:
            self.socket.bind(pipename)
        except zmq.ZMQError:
            print(
                "ZMQError in socket.bind(). Perhaps you're \
                using pipes on a non-local file system. See documentation of PrefetchDataZMQ for more information.")
            raise

        self._proc_ids = [u'{}'.format(k).encode('utf-8') for k in range(self.nr_proc)]
        worker_hwm = int(self._buffer_size * 2 // self.nr_proc)
        self._procs = [MultiProcessMapDataZMQ._Worker(
            self._proc_ids[k], self.map_func, pipename, worker_hwm)
            for k in range(self.nr_proc)]

        self.ds.reset_state()
        self._iter = self.ds.get_data()
        self._iter_worker = cycle(iter(self._proc_ids))

        for p in self._procs:
            p.deamon = True
            p.start()
        self._fill_buffer()     # pre-fill the bufer

    def reset_state(self):
        if self._reset_done:
            return
        self._reset_done = True

        # __del__ not guranteed to get called at exit
        atexit.register(del_weakref, weakref.ref(self))

        self._reset_once()  # build processes

    def _send(self, dp):
        # round-robin assignment
        worker = next(self._iter_worker)
        msg = [worker, dumps(dp)]
        self.socket.send_multipart(msg, copy=False)

    def _recv(self):
        msg = self.socket.recv_multipart(copy=False)
        dp = loads(msg[1].bytes)
        return dp

    def get_data(self):
        with self._guard, _zmq_catch_error('MultiProcessMapData'):
            if self._strict:
                for dp in self.get_data_strict():
                    yield dp
            else:
                for dp in self.get_data_non_strict():
                    yield dp

    def __del__(self):
        try:
            if not self._reset_done:
                return
            if not self.context.closed:
                self.socket.close(0)
                self.context.destroy(0)
            for x in self._procs:
                x.terminate()
                x.join(5)
            print_green("{} successfully cleaned-up.".format(type(self).__name__))
        except Exception:
            pass

def MultiProcessMapData(dp, map_func, nr_dpflows=0):
    if nr_dpflows == 0:
        dp = MapData(dp, map_func)
    else:
        dp = MultiProcessMapDataZMQ(dp, nr_dpflows, map_func)
    return dp

class BatchData(ProxyDataFlow):
    """
    Stack datapoints into batches.
    It produces datapoints of the same number of components as ``ds``, but
    each component has one new extra dimension of size ``batch_size``.
    The batch can be either a list of original components, or (by default)
    a numpy array of original components.
    """

    def __init__(self, ds, batch_size, use_list=False, use_concat=False):
        """
        Args:
            ds (DataFlow): When ``use_list=False``, the components of ``ds``
                must be either scalars or :class:`np.ndarray`, and have to be consistent in shapes.
            batch_size(int): batch size
            use_list (bool): if True, each component will contain a list
                of datapoints instead of an numpy array of an extra dimension.
                It's often used under the case that the data cannot be serialized or nparrayed.
            use_concat (int): 
                0: False
                1: True
                2: True and add batch_axis at the first axis (only work for numpy array)
                3: True and add batch_axis at the last axis (only work for numpy array)
        """
        self.ds = ds
        self.batch_size = int(batch_size)
        self.use_list = use_list
        self.use_concat = use_concat
        self.check_first = False

    def get_data(self):
        """
        Yields:
            Batched data by stacking each component on an extra 0th dimension.
        """
        holder = []
        for data in self.ds.get_data():
            holder.append(data)
            if len(holder) == self.batch_size:

                if not self.check_first:
                    size = len(holder[0])
                    if isinstance(self.use_list, list) or isinstance(self.use_list, tuple):
                        assert len(self.use_list) == size, 'use_list option should have same size of input data items.'
                    if isinstance(self.use_concat, list) or isinstance(self.use_concat, tuple):
                        assert len(self.use_concat) == size, 'use_concat option should have same size of input data items.'
                    self.check_first = True

                batch = BatchData._aggregate_batch(holder, self.use_list, self.use_concat)
                yield batch
                del batch[:]
                del holder[:]

    @staticmethod
    def _aggregate_batch(data_holder, use_list=False, use_concat=False):
        size = len(data_holder[0])
        if not (isinstance(use_list, list) or isinstance(use_list, tuple)):
            use_list = [use_list for i in range(size)]
        if not (isinstance(use_concat, list) or isinstance(use_concat, tuple)):
            use_concat = [use_concat for i in range(size)]

        result = []
        for k in range(size):
            if use_concat[k]:
                dt = data_holder[0][k]
                if isinstance(dt, list):
                    result.append(
                        list(chain(*[x[k] for x in data_holder])))
                else:
                    try:
                        if use_concat[k] is True or use_concat[k] == 1:
                            result.append(
                                np.concatenate([x[k] for x in data_holder], axis=0))
                        else:
                            if len(data_holder[0][k].shape) != 2:
                                raise ValueError('Cannot add batch axis in shape {}. It only supports 2-dim array.'.format(data_holder[0][k].shape))
                            if use_concat[k] == 2:
                                result.append(
                                    np.concatenate([np.pad(x[k], ((0, 0), (1, 0)), mode='constant', constant_values=i) for i,x in enumerate(data_holder)], axis=0))
                            elif use_concat[k] == 3:
                                result.append(
                                    np.concatenate([np.pad(x[k], ((0, 0), (0, 1)), mode='constant', constant_values=i) for i,x in enumerate(data_holder)], axis=0))
                            else:
                                raise ValueError('Unsupported type of attribute use_concat : {}'.format(use_concat[k]))
                    except Exception as e:  # noqa
                        print_yellow("Cannot concat batch data. Perhaps they are of inconsistent shape?")
                        if isinstance(dt, np.ndarray):
                            s = [x[k].shape for x in data_holder]
                            print_yellow("Shape of all arrays to be batched: {}".format(s))
                        try:
                            # open an ipython shell if possible
                            import IPython as IP; IP.embed()    # noqa
                        except ImportError:
                            pass
            else:
                if use_list[k]:
                    result.append(
                        [x[k] for x in data_holder])
                else:
                    dt = data_holder[0][k]
                    if type(dt) in [int, bool]:
                        tp = 'int32'
                    elif type(dt) == float:
                        tp = 'float32'
                    else:
                        try:
                            tp = np.asarray(dt).dtype
                        except AttributeError:
                            raise TypeError("Unsupported type to batch: {}".format(type(dt)))
                    try:
                        result.append(
                            np.asarray([x[k] for x in data_holder], dtype=tp))
                    except Exception as e:  # noqa
                        print_yellow("Cannot batch data. Perhaps they are of inconsistent shape?")
                        if isinstance(dt, np.ndarray):
                            s = [x[k].shape for x in data_holder]
                            print_yellow("Shape of all arrays to be batched: {}".format(s))
                        try:
                            # open an ipython shell if possible
                            import IPython as IP; IP.embed()    # noqa
                        except ImportError:
                            pass
        return result

class BatchDataNuscenes(ProxyDataFlow):
    """
    Stack datapoints into batches.
    It produces datapoints of the same number of components as ``ds``, but
    each component has one new extra dimension of size ``batch_size``.
    The batch can be either a list of original components, or (by default)
    a numpy array of original components.
    """

    def __init__(self, ds, batch_size, use_list=False, use_concat=False):
        """
        Args:
            ds (DataFlow): When ``use_list=False``, the components of ``ds``
                must be either scalars or :class:`np.ndarray`, and have to be consistent in shapes.
            batch_size(int): batch size
            use_list (bool): if True, each component will contain a list
                of datapoints instead of an numpy array of an extra dimension.
                It's often used under the case that the data cannot be serialized or nparrayed.
            use_concat (int): 
                0: False
                1: True
                2: True and add batch_axis at the first axis (only work for numpy array)
                3: True and add batch_axis at the last axis (only work for numpy array)
        """
        self.ds = ds
        self.batch_size = int(batch_size)
        self.use_list = use_list
        self.use_concat = use_concat
        self.check_first = False

    def get_data(self):
        """
        Yields:
            Batched data by stacking each component on an extra 0th dimension.
        """
        holder = []
        biggest_label_num_list = []
        for data in self.ds.get_data():
            biggest_label_num_list.append(data[0])
            holder.append(data[1:])
            if len(holder) == self.batch_size:
                # load done
                cur_biggest_label = np.max(biggest_label_num_list)
                if not self.check_first:
                    size = len(holder[0])
                    if isinstance(self.use_list, list) or isinstance(self.use_list, tuple):
                        assert len(self.use_list) == size, 'use_list option should have same size of input data items.'
                    if isinstance(self.use_concat, list) or isinstance(self.use_concat, tuple):
                        assert len(self.use_concat) == size, 'use_concat option should have same size of input data items.'
                    self.check_first = True

                batch = BatchDataNuscenes._aggregate_batch(holder, cur_biggest_label, self.use_list, self.use_concat)
                yield batch
                del batch[:]
                del holder[:]
                del biggest_label_num_list[:]

    @staticmethod
    def _aggregate_batch(data_holder, cur_biggest_label, use_list=False, use_concat=False):
        size = len(data_holder[0])
        if not (isinstance(use_list, list) or isinstance(use_list, tuple)):
            use_list = [use_list for i in range(size)]
        if not (isinstance(use_concat, list) or isinstance(use_concat, tuple)):
            use_concat = [use_concat for i in range(size)]

        result = []
        for k in range(size):
            if use_concat[k]:
                dt = data_holder[0][k]
                if isinstance(dt, list):
                    result.append(
                        list(chain(*[x[k] for x in data_holder])))
                else:
                    try:
                        if use_concat[k] is True or use_concat[k] == 1:
                            result.append(
                                np.concatenate([x[k] for x in data_holder], axis=0))
                        elif use_concat[k] == 3: # concatenate groundtruth
                            if len(data_holder[0][k].shape) == 3: # label_boxes_3d, label_anchors...
                                result.append(
                                    np.concatenate([np.pad(x[k], ((0, 0), (0, cur_biggest_label-x[k].shape[1]), (0, 0)), mode='constant', constant_values=0) for i,x in enumerate(data_holder)], axis=0))
                            elif len(data_holder[0][k].shape) == 2: # label_classes ...
                                result.append(
                                    np.concatenate([np.pad(x[k], ((0, 0), (0, cur_biggest_label-x[k].shape[1])), mode='constant', constant_values=0) for i,x in enumerate(data_holder)], axis=0))
                            else:
                                raise ValueError('Unsupported type of attribute use_concat : {}'.format(use_concat[k]))
                        else:
                            if len(data_holder[0][k].shape) == 2: # label_boxes_3d, label_anchors
                                if use_concat[k] == 2:
                                    result.append(
                                        np.stack([np.pad(x[k], ((0, cur_biggest_label-len(x[k])), (0, 0)), mode='constant', constant_values=0) for i,x in enumerate(data_holder)], axis=0))
                                else:
                                    raise ValueError('Unsupported type of attribute use_concat : {}'.format(use_concat[k]))
                            elif len(data_holder[0][k].shape) == 1: # label_class...
                                if use_concat[k] == 2:
                                    result.append(
                                        np.stack([np.pad(x[k], (0, cur_biggest_label-len(x[k])), mode='constant', constant_values=0) for i,x in enumerate(data_holder)], axis=0))
                                else:
                                    raise ValueError('Unsupported type of attribute use_concat : {}'.format(use_concat[k]))
                                
                    except Exception as e:  # noqa
                        print_yellow("Cannot concat batch data. Perhaps they are of inconsistent shape?")
                        if isinstance(dt, np.ndarray):
                            s = [x[k].shape for x in data_holder]
                            print_yellow("Shape of all arrays to be batched: {}".format(s))
                        try:
                            # open an ipython shell if possible
                            import IPython as IP; IP.embed()    # noqa
                        except ImportError:
                            pass
            else:
                if use_list[k]:
                    result.append(
                        [x[k] for x in data_holder])
                else:
                    dt = data_holder[0][k]
                    if type(dt) in [int, bool]:
                        tp = 'int32'
                    elif type(dt) == float:
                        tp = 'float32'
                    else:
                        try:
                            tp = np.asarray(dt).dtype
                        except AttributeError:
                            raise TypeError("Unsupported type to batch: {}".format(type(dt)))
                    try:
                        result.append(
                            np.asarray([x[k] for x in data_holder], dtype=tp))
                    except Exception as e:  # noqa
                        print_yellow("Cannot batch data. Perhaps they are of inconsistent shape?")
                        if isinstance(dt, np.ndarray):
                            s = [x[k].shape for x in data_holder]
                            print_yellow("Shape of all arrays to be batched: {}".format(s))
                        try:
                            # open an ipython shell if possible
                            import IPython as IP; IP.embed()    # noqa
                        except ImportError:
                            pass
        return result

class StoppableThread(threading.Thread):
    """
    A thread that has a 'stop' event.
    """

    def __init__(self, evt=None):
        """
        Args:
            evt(threading.Event): if None, will create one.
        """
        super(StoppableThread, self).__init__()
        if evt is None:
            evt = threading.Event()
        self._stop_evt = evt

    def stop(self):
        """ Stop the thread"""
        self._stop_evt.set()

    def stopped(self):
        """
        Returns:
            bool: whether the thread is stopped or not
        """
        return self._stop_evt.isSet()

    def queue_put_stoppable(self, q, obj):
        """ Put obj to queue, but will give up when the thread is stopped"""
        while not self.stopped():
            try:
                q.put(obj, timeout=5)
                break
            except queue.Full:
                pass

    def queue_get_stoppable(self, q):
        """ Take obj from queue, but will give up when the thread is stopped"""
        while not self.stopped():
            try:
                return q.get(timeout=5)
            except queue.Empty:
                pass

class MultiThreadMapData(_ParallelMapData):
    """
    Same as :class:`MapData`, but start threads to run the mapping function.
    This is useful when the mapping function is the bottleneck, but you don't
    want to start processes for the entire dataflow pipeline.

    Note:
        1. There is tiny communication overhead with threads, but you
           should avoid starting many threads in your main process to reduce GIL contention.

           The threads will only start in the process which calls :meth:`reset_state()`.
           Therefore you can use ``PrefetchDataZMQ(MultiThreadMapData(...), 1)``
           to reduce GIL contention.

        2. Threads run in parallel and can take different time to run the
           mapping function. Therefore the order of datapoints won't be
           preserved, and datapoints from one pass of `df.get_data()` might get
           mixed with datapoints from the next pass.

           You can use **strict mode**, where `MultiThreadMapData.get_data()`
           is guranteed to produce the exact set which `df.get_data()`
           produces. Although the order of data still isn't preserved.
    """
    class _Worker(StoppableThread):
        def __init__(self, inq, outq, evt, map_func):
            super(MultiThreadMapData._Worker, self).__init__(evt)
            self.inq = inq
            self.outq = outq
            self.func = map_func
            self.daemon = True

        def run(self):
            try:
                while True:
                    dp = loads(self.queue_get_stoppable(self.inq))
                    if self.stopped():
                        return
                    # cannot ignore None here. will lead to unsynced send/recv
                    obj = self.func(dp)
                    self.queue_put_stoppable(self.outq, dumps(obj))
            except Exception:
                if self.stopped():
                    pass        # skip duplicated error messages
                else:
                    raise
            finally:
                self.stop()

    def __init__(self, ds, nr_thread, map_func, buffer_size=200, strict=False):
        """
        Args:
            ds (DataFlow): the dataflow to map
            nr_thread (int): number of threads to use
            map_func (callable): datapoint -> datapoint | None
            buffer_size (int): number of datapoints in the buffer
            strict (bool): use "strict mode", see notes above.
        """
        super(MultiThreadMapData, self).__init__(ds, buffer_size)

        self._strict = strict
        self.nr_thread = nr_thread
        self.map_func = map_func
        self._threads = []
        self._evt = None

    def reset_state(self):
        super(MultiThreadMapData, self).reset_state()
        if self._threads:
            self._threads[0].stop()
            for t in self._threads:
                t.join()

        self._in_queue = queue.Queue()
        self._out_queue = queue.Queue()
        self._evt = threading.Event()
        self._threads = [MultiThreadMapData._Worker(
            self._in_queue, self._out_queue, self._evt, self.map_func)
            for _ in range(self.nr_thread)]
        for t in self._threads:
            t.start()

        self._iter = self.ds.get_data()
        self._guard = DataFlowReentrantGuard()

        # Call once at the beginning, to ensure inq+outq has a total of buffer_size elements
        self._fill_buffer()

    def _recv(self):
        x = loads(self._out_queue.get())
        return x

    def _send(self, dp):
        self._in_queue.put(dumps(dp))

    def get_data(self):
        with self._guard:
            if self._strict:
                for dp in self.get_data_strict():
                    yield dp
            else:
                for dp in self.get_data_non_strict():
                    yield dp

    def __del__(self):
        if self._evt is not None:
            self._evt.set()
        for p in self._threads:
            p.stop()
            p.join(timeout=5.0)
            # if p.is_alive():
            #     logger.warn("Cannot join thread {}.".format(p.name))
