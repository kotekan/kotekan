import sys
import h5py as h5
import numpy as np

def print_attrs(name, obj):

    print("Group name: {}".format(name))
    for key, val in obj.attrs.iteritems():
        print("    {}: {}").format(key, val)

def main():
    in_file = sys.argv[1]

    print("Reading file: {}".format(in_file))

    f = h5.File(in_file, "r")

    print("\nAttributes")
    print("----------")
    for a in f.attrs:
        print("{}: {}".format(a, f.attrs[a]))

    print("\nDatasets")
    print("-------")
    group = f["/"]
    for dset in group:
        print("{}: {}".format(dset, group[dset]))

    group = f["/index_map"]
    for dset in group:
        print("{}: {}".format(dset, group[dset]))

    #f.visititems(print_attrs)


if __name__ == "__main__":
    main()
