import os
import numpy as np
from collections import OrderedDict


class Cache:
    def __init__(self, cache_dir_path=None, max_size=100, sparsity=False, keep_order=False):
        self.cache_dir_path = cache_dir_path
        self.max_size = max_size #todo: calculte the max memory size
        self.sparsity = sparsity
        self.keep_order = keep_order
        self._memory_cache = OrderedDict()

        if cache_dir_path and os.path.exists(cache_dir_path):
            cache_files = [f for f in os.listdir(cache_dir_path) if
                           os.path.isfile(os.path.join(cache_dir_path, f))]
            for i, cache_file in enumerate(cache_files):
                data = np.load(os.path.join(cache_dir_path, cache_file))
                self._memory_cache.update(data)
                if i >= self.max_size:
                    break

    def read(self, id):
        if self.cache_dir_path is None:
            return None
        if str(id) in self._memory_cache:
            if self.keep_order:
                value = self._memory_cache.pop(str(id))
                self._memory_cache[str(id)] = value
            else:
                value = self._memory_cache[str(id)]
            return value
        elif self.cache_dir_path and os.path.exists(self._get_cache_file_path(id)):
            with np.load(self._get_cache_file_path(id)) as data:
                self._memory_cache.update(data)
                value = data[str(id)]
                self._memory_cache[str(id)] = value
                if len(self._memory_cache) > self.max_size:
                    self._memory_cache.popitem(last=False)
                return value
        else:
            return None

    def write(self, id, data):
        if self.cache_dir_path is None:
            return
        self._memory_cache[str(id)] = data

        if self.cache_dir_path:
            cache_file_path = self._get_cache_file_path(id)
            np.savez_compressed(cache_file_path, **{str(id): data})

        if len(self._memory_cache) > self.max_size:
            self._memory_cache.popitem(last=False)

    def clear_memory_cache(self):
        self._memory_cache = OrderedDict()

    def _get_cache_file_path(self, id):
        return os.path.join(self.cache_dir_path, f"{id}.npz")
