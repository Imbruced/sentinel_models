
class classproperty(object):

    def __init__(self, f):
        self.f = f

    def __get__(self, obj, owner):
        return self.f(owner)


def lazy_property(fn):
    attr_name = '__' + fn.__name__

    @property
    def wrapper(self):
        if not hasattr(self, attr_name):
            setattr(self, attr_name, fn(self))
        return getattr(self, attr_name)

    return wrapper