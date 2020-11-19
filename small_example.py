from abc import ABCMeta, abstractmethod
from collections import defaultdict
from enum import Enum
import errno
from functools import reduce
from future.utils import with_metaclass
import importlib
import os
import shutil

from faker.providers.person.en import Provider
import numpy as np
import pandas as pd
import sqlalchemy as sa
import taskflow.engines
from taskflow.patterns import linear_flow as lf
from taskflow import task


def create_directory_if_does_not_exists(dir_path):
    try:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

def load_module_by_name(module_name):
    mdl = importlib.import_module(module_name)
    return mdl

def load_class(full_class_name):
    module_parts = full_class_name.split(".")
    klass_name = module_parts[-1]
    module_path = ".".join(module_parts[0:-1])
    module = load_module_by_name(module_path)
    klass = getattr(module, klass_name)
    return klass

__iomanager = None
def get_io_manager():
    global __iomanager
    __iomanager = __iomanager or DataSetIOManager(DataSetIOHandlerType.memory)
    return __iomanager

class DataSetIOHandlerType(Enum):
    memory = 1
    h5 = 2
    sql = 3

class DataSetIOHandler(with_metaclass(ABCMeta, object)):
    store_type = None
    uri_protocol_separator = "://"

    @classmethod
    def get_uri_prefix(cls, uri):
        return DataSetIOHandlerType[uri.split(cls.uri_protocol_separator)[0]]

    @classmethod
    def get_uri_suffix(cls, uri):
        return uri.split(cls.uri_protocol_separator)[1]

    @abstractmethod
    def get_storage_key(self, dataset_name):
        raise NotImplementedError()

    def get_uri(self, dataset_name):
        return "{0}://{1}".format(self.store_type._name_, self.get_storage_key(dataset_name))

    @abstractmethod
    def store(self, dataset_name, dataset):
        raise NotImplementedError()

    @abstractmethod
    def retrieve(self, dataset_uri):
        raise NotImplementedError()

    @abstractmethod
    def delete(self, dataset_uri):
        raise NotImplementedError()

    @abstractmethod
    def clear(self):
        raise NotImplementedError()

class MemoryDataSetIOHandler(DataSetIOHandler):
    store_type = DataSetIOHandlerType.memory

    def __init__(self):
        self.data_items = dict()

    def get_storage_key(self, dataset_name):
        return dataset_name

    def store(self, dataset_name, dataset):
        key = self.get_uri(dataset_name)
        self.data_items[key] = dataset
        return key

    def retrieve(self, dataset_uri):
        data = self.data_items[dataset_uri]
        return data

    def delete(self, dataset_uri):
        del self.data_items[dataset_uri]

    def clear(self):
        self.data_items = dict()

class H5DataSetIOHandler(DataSetIOHandler):
    store_type = DataSetIOHandlerType.h5

    def __init__(self, base_location, complib="blosc", complevel=9):
        self.base_location = base_location
        self.complib = complib
        self.complevel = complevel
        create_directory_if_does_not_exists(self.base_location)

    def get_storage_key(self, dataset_name):
        return os.path.join(self.base_location, dataset_name+".h5")

    def store(self, dataset_name, dataset):
        key = self.get_uri(dataset_name)
        file_loc = self.get_uri_suffix(key)
        with pd.HDFStore(file_loc, "w", complib=self.complib, complevel=self.complevel) as store:
            store["data"] = dataset
        return key

    def retrieve(self, dataset_uri):
        file_loc = self.get_uri_suffix(dataset_uri)
        with pd.HDFStore(file_loc, "r", complib=self.complib, complevel=self.complevel) as store:
            dataset = store["data"]
        return dataset

    def delete(self, dataset_uri):
        file_loc = self.get_uri_suffix(dataset_uri)
        os.remove(file_loc)

    def clear(self):
        shutil.rmtree(self.base_location, ignore_errors=False, onerror=None)

class SQLDataSetIOHandler(DataSetIOHandler):
    store_type = DataSetIOHandlerType.sql

    def __init__(self, connection_str):
        self.connection_str = connection_str

    def get_storage_key(self, dataset_name):
        return dataset_name

    def store(self, dataset_name, dataset):
        pass

    def retrieve(self, dataset_uri):
        pass

    def delete(self, dataset_uri):
        pass

    def clear(self):
        pass


class DataSetIOManager(object):
    def __init__(self, default_handler_type):
        self.iohanders = dict()
        self.iohanders[MemoryDataSetIOHandler.store_type] = MemoryDataSetIOHandler()
        self.iohanders[H5DataSetIOHandler.store_type] = H5DataSetIOHandler(r"C:\TAP\temp\product")
        self.default_handler_type = default_handler_type
        self.results_uris = dict()

    def get_default_handler(self):
        return self.default_handler_type

    def store_results(self, dataset_name, dataset, handler_type=None, alias=None):
        handler = self.iohanders[handler_type or self.default_handler_type]
        result = handler.store(dataset_name, dataset)
        self.results_uris[dataset_name] = result
        if alias is not None:
            if alias in self.results_uris:
                raise RuntimeError("Attempting to store a dataset result with alias {0} when a result with the same alias exists".format(alias))
            self.results_uris[alias] = result
        return result

    def load_result(self, dataset_uri):
        handler_type = DataSetIOHandler.get_uri_prefix(dataset_uri)
        handler = self.iohanders[handler_type]
        dataset = handler.retrieve(dataset_uri)
        return dataset

    def load_result_by_dataset_name(self, dataset_name):
        if dataset_name not in self.results_uris:
            raise KeyError("Dataset {0} is not present on the set of available results".format(dataset_name))
        result = self.load_result(self.results_uris[dataset_name])
        return result

class TAPDataPipelineTask(task.Task):
    def __init__(self, storage_type=None, name=None, provides=None, requires=None, auto_extract=True,
                 rebind=None, inject=None, ignore_list=None, revert_rebind=None, revert_requires=None):
        self.storage_type = storage_type or get_io_manager().get_default_handler()
        task.Task.__init__(self, name, provides, requires, auto_extract, rebind, inject, ignore_list, revert_rebind, revert_requires)

    def do_task(self, *args, **kwargs):
        raise NotImplementedError()

    def _unpack_inputs(self, **kwargs):
        iomanager = get_io_manager()
        inputs = {k:iomanager.load_result(kwargs[k]) for k in self.requires}
        return inputs

    def _pack_result(self, result):
        iomanager = get_io_manager()
        if len(self.provides)>1 and isinstance(result, dict):
            stored_result = {k:iomanager.store_results(k, result[k], handler_type=self.storage_type)}
            result = stored_result
        elif len(self.provides)==1:
            key = list(self.provides)[0]
            result = {key: iomanager.store_results(key, result, handler_type=self.storage_type)}
        return result

    def execute(self, *args, **kwargs):
        inputs = self._unpack_inputs(**kwargs)
        result = self.do_task(**inputs)
        packed_result = self._pack_result(result)
        return packed_result

class ColumnBuilder(TAPDataPipelineTask):
    def __init__(self, colname=None, storage_type=None):
        colname = colname or self.__class__.__name__
        TAPDataPipelineTask.__init__(self, provides=set([colname]), requires="raw_df", storage_type=storage_type)

    def _pack_result(self, result):
        result.name = list(self.provides)[0]
        return TAPDataPipelineTask._pack_result(self, result)

class MergeColumns(TAPDataPipelineTask):
    def __init__(self, flow):
        provides_flow_data = [k[0].provides for k in flow.iter_nodes() if isinstance(k[0], ColumnBuilder)]
        provides_flow_data = list(reduce(lambda a,b:a.union(b), provides_flow_data))
        TAPDataPipelineTask.__init__(self, requires=provides_flow_data, provides=set([flow.name]),
                                     storage_type=flow.get_io_handler_type())

    def do_task(self, *args, **kwargs):
        all_series = [kwargs[k] for k in self.requires]
        result = pd.concat(all_series, axis=1)
        return result

class DatasetBuilder(lf.Flow):
    def __init__(self, name, retry=None, storage_type=None):
        self.storage_type=storage_type
        lf.Flow.__init__(self, name, retry)

    def get_io_handler_type(self):
        return self.storage_type

    def add(self, *items):
        incorrect_tasks = [k for k in items if not isinstance(k, ColumnBuilder)]
        if incorrect_tasks:
            raise RuntimeError("{0}.add only accepts instances of ColumnBuilder classes. Received {1}".format(self.__class__.__name__, incorrect_tasks))
        lf.Flow.add(self, *items)

    def compile(self):
        merge_task = MergeColumns(self)
        lf.Flow.add(self, merge_task)

class RandomData(TAPDataPipelineTask):
    PROVIDES = "raw_df"

    def __init__(self, n_rows):
        TAPDataPipelineTask.__init__(self, provides=self.__class__.PROVIDES)
        self.n_rows = n_rows

    def random_names(self, name_type):
        """
        Generate n-length ndarray of person names.
        name_type: a string, either first_names or last_names
        """
        name_type
        names = getattr(Provider, name_type)
        return np.random.choice(names, size=self.n_rows)

    def random_genders(self, p=None):
        """Generate n-length ndarray of genders."""
        p = p or (0.49, 0.49, 0.01, 0.01)
        gender = ("M", "F", "O", "")
        return np.random.choice(gender, size=self.n_rows, p=p)

    def random_dates(self):
        """
        Generate random dates within range between start and end.
        Adapted from: https://stackoverflow.com/a/50668285
        """
        # Unix timestamp is in nanoseconds by default, so divide it by
        # 24*60*60*10**9 to convert to days.
        start = pd.to_datetime('1940-01-01')
        end = pd.to_datetime('2008-01-01')
        divide_by = 24 * 60 * 60 * 10**9
        start_u = start.value // divide_by
        end_u = end.value // divide_by
        return pd.to_datetime(np.random.randint(start_u, end_u, self.n_rows), unit="D")

    def do_task(self, *args, **kwargs):
        df = pd.DataFrame(columns=['First', 'Last', 'Gender', 'Birthdate'])
        df['First'] = self.random_names('first_names')
        df['Last'] = self.random_names('last_names')
        df['Gender'] = self.random_genders()
        df['Birthdate'] = self.random_dates()
        return df

class CountGenderByDate(ColumnBuilder):
    def do_task(self, *args, **kwargs):
        raw_df = kwargs["raw_df"]
        result = raw_df.groupby("Birthdate")["Gender"].agg(len)
        return result

class CountFirstNameInitialByDate(ColumnBuilder):
    def do_task(self, *args, **kwargs):
        raw_df = kwargs["raw_df"]
        result = raw_df.groupby("Birthdate")["First"].agg(len)
        return result

class CountLastNameInitialByDate(ColumnBuilder):
    def do_task(self, *args, **kwargs):
        raw_df = kwargs["raw_df"]
        result = raw_df.groupby("Birthdate")["Last"].agg(len)
        return result

class ModelDataBatchType(Enum):
    Training = 1
    Validation = 2
    Test = 3

class ModelDataGenerator(TAPDataPipelineTask):
    def __init__(self, generator_name, input_column_groups, output_column_groups, batch_size, n_batches=None, data_splits_pctg=None,
                 data_splits_int=None, **kwargs):
        self.input_column_groups = input_column_groups
        self.output_column_groups = output_column_groups
        self._all_columns_groups = self.input_column_groups + self.output_column_groups
        self.batch_size = batch_size
        self.n_batches = n_batches
        self.data_splits_pctg = data_splits_pctg
        self.data_splits_int = data_splits_int
        kwargs["provides"] = generator_name
        kwargs["requires"] = self._parse_columns_groups_into_requires()
        TAPDataPipelineTask.__init__(self, **kwargs)

    def _parse_columns_groups_into_requires(self):
        requires_inputs = set()
        for col_group in self._all_columns_groups:
            for col in col_group:
                requires_inputs.add(col.split(".")[0])
        requires_inputs = list(requires_inputs)
        return requires_inputs

    def _get_raw_data_batch(self, batch_type):
        raise NotImplementedError()

    def _initialize_data_source(self, **kwargs):
        raise NotImplementedError()

    def _split_data_batch_in_groups(self, raw_data_batch):
        inputs = [raw_data_batch[cols] for cols in self.input_column_groups]
        outputs = [raw_data_batch[cols] for cols in self.input_column_groups]
        return (inputs, outputs)

    def generate_train_data(self):
        while True:
            data = self._get_raw_data_batch(ModelDataBatchType.Training)
            inputs, outputs = self._split_data_batch_in_groups(data)
            yield (inputs, outputs)
        pass

    def generate_validation_data(self):
        while True:
            data = self._get_raw_data_batch(ModelDataBatchType.Validation)
            inputs, outputs = self._split_data_batch_in_groups(data)
            yield (inputs, outputs)
        pass

    def generate_test_data(self):
        while True:
            data = self._get_raw_data_batch(ModelDataBatchType.Test)
            inputs, outputs = self._split_data_batch_in_groups(data)
            yield (inputs, outputs)
        pass

    def execute(self, **kwargs):
        self._initialize_data_source(**kwargs)
        return self

class InMemoryModelDataGenerator(ModelDataGenerator):
    def __init__(self, generator_name, input_column_groups, output_column_groups, batch_size,
                 n_batches=None, data_splits_pctg=None, data_splits_int=None, **kwargs):
        self.raw_data = None
        self.n_rows = None
        ModelDataGenerator.__init__(self, generator_name, input_column_groups, output_column_groups, batch_size,
                                    n_batches, data_splits_pctg, data_splits_int, **kwargs)

    def _get_raw_data_batch(self, batch_type):
        return self.raw_data[0:self.batch_size]

    def _initialize_data_source(self, **kwargs):
        io_manager = get_io_manager()
        raw_columns = []
        for ds_name in self.requires:
            dataset = io_manager.load_result(kwargs[ds_name])
            for col_group in self._all_columns_groups:
                required_cols = [col for col in col_group if col.split(".")[0]==ds_name]
                for col in required_cols:
                    selected_col = dataset[col.split(".")[1]]
                    selected_col.name = col
                    raw_columns.append(selected_col)
        self.raw_data = pd.concat(raw_columns, axis=1)
        self.n_rows = len(self.raw_data)

class TAPModelTrainingTask(TAPDataPipelineTask):
    def __init__(self, model_class, model_kwargs, data_generator_provider, **kwargs):
        klass = load_class(model_class)
        self.model_instance = klass(**model_kwargs)
        self.serialized_model = None
        kwargs["provides"] = kwargs.get("provides", self.__class__.__name__)
        kwargs["requires"] = data_generator_provider.provides
        TAPDataPipelineTask.__init__(self, **kwargs)

    def _do_training(self):
        raise NotImplementedError("TAPModelTrainingTask._do_training must be overriden in derived class")

    def _serialize_model(self):
        raise NotImplementedError("TAPModelTrainingTask._serialize_model must be overriden in derived class")

    def execute(self, **kwargs):
        data_generator = kwargs[list(self.requires)[0]]
        sample_data = data_generator.generate_train_data()
        return True


class SklearnModel(TAPModelTrainingTask):
    def __init__(self, model_class, model_kwargs, data_generator_provider, **kwargs):
        TAPModelTrainingTask.__init__(self, model_class, model_kwargs, data_generator_provider, **kwargs)

    def _do_training(self):
        raise NotImplementedError("TAPModelTrainingTask._do_training must be overriden in derived class")

    def _serialize_model(self):
        raise NotImplementedError("TAPModelTrainingTask._serialize_model must be overriden in derived class")


if __name__=="__main__":
    rand_df_ref = RandomData(10000).execute()

    feature_data = DatasetBuilder("counts_per_date", storage_type=DataSetIOHandlerType.h5)
    feature_data.add(CountGenderByDate(),
                     CountFirstNameInitialByDate(storage_type=DataSetIOHandlerType.h5),
                     #CountFirstNameInitialByDate(),
                     CountLastNameInitialByDate())
    feature_data.compile()

    clustering_data_generator = InMemoryModelDataGenerator("clustering_data", [["counts_per_date.CountGenderByDate",
                                                                               "counts_per_date.CountFirstNameInitialByDate"]], [], 20)
    clustering = SklearnModel("sklearn.cluster.KMeans",
                              {"n_clusters": 2, "max_iter": 100},
                              clustering_data_generator)
    full_flow = lf.Flow("tap_test")
    full_flow.add(feature_data, clustering_data_generator, clustering)

    engine = taskflow.engines.load(full_flow, store=rand_df_ref)
    engine.run()

    print("Done")