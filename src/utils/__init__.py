import os
import cloudpickle
import hashlib


def cache_dataset(cache_path="/tmp/"):
    def cache_dataset_decorator(fn):
        def cache(*args, **kwargs):
            params_str = str(args + tuple(sorted(kwargs.items())))
            corpus_name = 'corpus.{}.data'.format(
                hashlib.md5(params_str.encode()).hexdigest())

            if os.path.exists(cache_path + corpus_name):
                with open(cache_path + corpus_name, "rb") as f:
                    cache = cloudpickle.load(f)
                return cache

            output = fn(*args, **kwargs)
            cache_obj = {}
            if isinstance(output, tuple):
                for idx, out in enumerate(output):
                    cache_obj["obj_{}".format(idx)] = out
            elif isinstance(output, dict):
                for key in output.keys():
                    cache_obj[key] = output[key]
            else:
                cache_obj["object"] = output

            with open(cache_path + corpus_name, "wb") as f:
                cloudpickle.dump(cache_obj, f)

        return cache

    return cache_dataset_decorator
