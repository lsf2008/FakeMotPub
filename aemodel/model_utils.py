import importlib
import sys
class Loader(object):
    def __init__(self, module_name):
        self.module_name = module_name
        self.module = None
        self.load_module()
    def load_module(self):
        try:
            self.module = importlib.__import__(self.module_name)
        except:
            print(f'Module {self.module_name} not find')
            sys.exit(1)
    def get_instance(self, class_name, *args, **kwargs):
        try:
            class_obj = getattr(self.module_name, class_name)
            instance = class_obj(*args, **kwargs)
            return instance
        except AttributeError:
            print(f'class {class_name} not find in module {self.module_name}')
            return None

