import json


class Config:
    """Config class which contains data, train and model hyperparameters"""

    def __init__(self, data=None, parameters=None, output=None):
        self.data = data
        self.parameters = parameters
        self.output = output

    @classmethod  # using config to define constructor of the class
    def from_json(cls, cfg):
        """Creates config from json"""
        params = json.loads(json.dumps(cfg), object_hook=HelperDict)
        # init all class instance with data and train attributes
        return cls(params.data, params.parameters, params.output)


class HelperDict(object):
    """Helper class to convert json into Python object"""

    def __init__(self, dict_):
        self.__dict__.update(dict_)

    def __getitem__(self, key):
        return getattr(self, key)

    def get(self, key, default=None):
        return getattr(self, key, default)
