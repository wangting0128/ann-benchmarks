from __future__ import absolute_import
import numpy
import sklearn.preprocessing
from pymilvus import DataType, DefaultConfig, CollectionSchema, FieldSchema, connections, Collection
from ann_benchmarks.algorithms.base import BaseANN


class Milvus(BaseANN):
    def __init__(self, metric, dim, index_param):
        self._index_type = "HNSW"

        self._metric = metric
        self._dim = dim
        self._metric_type = "IP" if self._metric == "angular" else "L2"
        self._index_params = {"index_type": self._index_type, "metric_type": self._metric_type, "params": index_param}

        self._ef = None  # search param
        self._search_param = {}
        self.connection = connections.connect(alias=DefaultConfig.DEFAULT_USING, host=DefaultConfig.DEFAULT_HOST,
                                              port=DefaultConfig.DEFAULT_PORT)
        self.collection = None
        self.collection_name = "milvus_benchmark"
        self.anns_field = "float_vector"

    def fit(self, X):
        if self._metric == 'angular':
            X = sklearn.preprocessing.normalize(X, axis=1, norm='l2')

        def iter_data(v, batch=50000):
            train_len = len(v)
            all_iter = [batch for i in range(batch, train_len, batch)]
            if train_len % batch > 0:
                all_iter += (train_len % batch,)

            _start = 0
            for i in all_iter:
                _end = _start + i
                yield [[d for d in range(_start, _end)], v[_start:_end]]
                _start = _end

        # create collection
        fields = [FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
                  FieldSchema(name=self.anns_field, dtype=DataType.FLOAT_VECTOR, dim=self._dim)]
        schema = CollectionSchema(fields=fields)
        self.collection = Collection(self.collection_name, schema=schema)

        entities = iter_data(X.tolist())
        for ent in entities:
            self.collection.insert(ent)

        self.collection.flush()
        self.collection.create_index(field_name=self.anns_field, index_params=self._index_params)
        self.collection.load()

    def set_query_arguments(self, ef):
        self._ef = ef
        self._search_param = {"params": {"ef": self._ef}, "metric_type": self._metric_type}

    def query(self, v, n):
        if self._metric == 'angular':
            v /= numpy.linalg.norm(v)
        res = self.collection.search(data=[v.tolist()], anns_field=self.anns_field, param=self._search_param,
                                     limit=n)
        return res[0].ids

    def __str__(self):
        return f"Milvus({self._index_params}, search_ef={self._ef})"
