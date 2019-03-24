import logging

logging.basicConfig(level="INFO")


class NumberDescriptor(object):

    def snoop_name(self, owner):
        logging.info(dir(owner))
        for attr in dir(owner):
            if getattr(owner, attr) is self:
                return attr

    def __get__(self, instance, owner):
        if instance is None:
            return self
        name = self.snoop_name(owner)
        return getattr(instance, '_'+name)

    def __set__(self, instance, value):
        name = self.snoop_name(type(instance))
        setattr(instance, '_' + name, int(value))


class A:
    __prop = NumberDescriptor()

    def __init__(self, value):
        self.__prop = value

    @property
    def prop(self):
        return self.__prop


a = A(10)
b = A(20)

logging.info(a.prop)
logging.info(b.prop)
