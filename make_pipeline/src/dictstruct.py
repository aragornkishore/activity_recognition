#!/usr/bin/env python
#-*- coding: utf-8 -*-

import commontools


def pprint_dictstruct(d, indentspaces=4, indentlevel=1):
    return commontools.pprint_sorteddict(dictstruct_to_dict(d))

def saferepr_dictstruct(d, indentspaces=4, indentlevel=1):
    return commontools.saferepr_sorteddict(dictstruct_to_dict(d))

def dictstruct_to_dict(d):
    if isinstance(d, dict):
        items = d.items
    elif isinstance(d, DictStruct):
        items = d.__dict__.items
    else:
        raise ValueError('d not of type dict or DictStruct')
    d_dict = {}
    for k, v in items():
        if isinstance(v, DictStruct) or isinstance(v, dict):
            d_dict[k] = dictstruct_to_dict(v)
        else:
            d_dict[k] = v
    return d_dict

class DictStruct(object):
    """Recursive class for building and representing objects with.
    Warning: Can only be used for simple Python datatypes! This class is mainly
             used to manage human-readable config files with hierarchical
             settings.
    """
    def __init__(self, obj=None):
        """Constructor, that recursively converts dictionaries to member vars
        Args:
            obj: an object, that has the method iteritems(), e.g. a dictionary
                 or a string, containing the path to a file with a valid
                 DictStruct.__repr__() string in it.
        """
        if isinstance(obj, str):
            self.load(obj)
        elif isinstance(obj, dict) or isinstance(obj, DictStruct):
            self.fromdict(obj)
        else:
            self.fromdict({})

    def fromdict(self, d):
        for k, v in d.iteritems():
            if isinstance(v, dict) or isinstance(v, DictStruct):
                setattr(self, k, DictStruct(v))
            else:
                setattr(self, k, v)

    def iteritems(self):
        """Wrapper for the dictionary iteritems function of the member dict
        """
        return self.__dict__.iteritems()

    def __getitem__(self, val):
        """Wrapper for the dictionary getter function of the member dict
        Args:
            val: name of a member variable
        """
        return self.__dict__[val]

    def __contains__(self, item):
        return item in self.__dict__

    def __repr__(self):
        """Generates Represention string, that can be used in the constructor
        Returns: Representation string that is the same as for dictionary
        """
        return saferepr_dictstruct(self)

    def __str__(self):
        """Conversion to a string
        """
        return pprint_dictstruct(self)

    def todict(self):
        """Converts the dictstruct to a dictionary
        """
        return dictstruct_to_dict(self)

    def dump(self, fname):
        """Dumps the dictstruct contents to a file that can be loaded
        Args:
            fname: filepath to the file
        """
        with open(fname, 'w') as f:
            f.write(self.__repr__())

    def load(self, fname):
        """Loads dictstruct contents from a file created by DictStruct.dump()
        Args:
            fname: filepath to the file
        """
        with open(fname, 'r') as f:
            self.fromdict(eval(f.read()))

# vim: set ts=4 sw=4 sts=4 expandtab:
