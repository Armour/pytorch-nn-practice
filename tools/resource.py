# py2 and py3 compatibility
from __future__ import print_function, absolute_import, division

import sys
import json
import random

# TODO: fix relavant imports
# pycharm workaround

from .gpustat import GPUStatCollection

class gpuManager():
    def __init__(self):
        self.update()

    def update(self):
        try:
            self.monitor = GPUStatCollection.new_query()
            self.info = self.monitor.jsonify()
            self.gpu_info = self.info['gpus']
            self.gpu_nums = len(self.gpu_info)
            self.memory_usage = [
                [each["memory.used"], each["memory.total"]] for each in self.gpu_info
            ]
        except Exception:
            sys.stderr.write('Error on querying NVIDIA devices\n')
            sys.stderr.write('''Did you have 'nvidia-smi' installed properly? \n''')
            sys.exit(1)

    # TODO: add fan_limit=None, volatile_limit=None,
    def allocate(self, n, memory_limit=0.2, shuffle=True):
        """
        :param n: the gpus that will be requested
        :param limit: only memory
        :return:
            a list of avaliable gpus
        """
        self.update()
        if n > self.gpu_nums:
            # sys.stderr.write("Not enough devices on local machine\n")
            return None

        choices = []

        for i in range(self.gpu_nums):
            used = self.memory_usage[i][0]
            total = self.memory_usage[i][1]

            if used / total < memory_limit:
                choices.append(i)

        if n > len(choices):
            # sys.stderr.write("Fail to find %s free GPUs on local machine\n" % n)
            return None

        if shuffle:
            random.shuffle(choices)
            return choices[:n]

    @staticmethod
    def date_handler(obj):
        if hasattr(obj, 'isoformat'):
            return obj.isoformat()
        else:
            raise TypeError(type(obj))




if __name__ == "__main__":
    admin = gpuManager()

    print(admin.allocate(2, memory_limit=0.1))