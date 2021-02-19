# Copyright (c) 2018-2019, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

def init():
    global async_evaluator
    global current_tag
    async_evaluator = AsyncEvaluator(1)
    current_tag = 0

def get_evaluator():
    global async_evaluator
    return async_evaluator

def get_tag():
    global current_tag
    return current_tag

def set_epoch_tag(epoch):
    global current_tag
    current_tag = epoch

class AsyncEvaluator():
    """
    Creates a threaded evaluator for a given device.
    If device == None then the current active device is used
    """
    def __init__(self, num_threads=1, device=None):
        self.num_threads = num_threads
        # self.pool = ThreadPoolExecutor(num_threads)
        self.pool = ProcessPoolExecutor(num_threads)

        self.events = {}

    def __del__(self):
        for t, e in self.events.items():
           e.cancel()

    # submit given function and its arguments with an
    # associated tag for bookkkeeping
    def submit_task(self, tag, fn, *args, **kwargs):

        # launch work
        e = self.pool.submit(fn, *args, **kwargs)

        # record work
        self.events[tag] = e

    # check if a task has completed
    def task_done(self, tag):
        return self.events[tag].done()

    # get the result of a task:
    # Note: will block until completed
    def task_result(self, tag):
        return self.events[tag].result(timeout=None)

    # Get all currently finished tasks in a dict of
    # { tag : result }
    def finished_tasks(self):
        ret = {}
        to_remove = []
        # Check all existing tasks
        for t in self.events.keys():
            done = self.events[t].done()

            if done:
                ret[t] = self.task_result(t)
                to_remove.append(t)

        # As soon as a task is finished we want to remove it
        for t in to_remove:
            self.task_remove(t)

        return ret

    # remove a task from the outstanding list
    # Note: will cancel task if not completed
    def task_remove(self, tag):
        done = self.events[tag].done()

        # cancel task if necessary
        if not done:
            self.events[tag].cancel()

        # remove the entry
        del self.events[tag]

    # return list of tags outstanding
    def task_tags(self):
        return self.events.keys()

    # wait for everything to finish
    def wait_all_tasks(self):
        for t in self.events.keys():
            y = self.task_result(t)
            print('task {} finished'.format(t))

