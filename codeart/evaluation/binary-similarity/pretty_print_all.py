import os
import glob
import sys

benchmarks = [
    "coreutilsh",
    "binutilsh",
    "libcurlh",
    "libmagickh",
    "opensslh",
    "libsqlh",
    "puttyh",
]

pool_sizes = [
  32, 50, 100, 200, 300, 500
]


print("Pretty print for ", sys.argv[1])

prefix = sys.argv[1]
if prefix.endswith("-"):
    prefix = prefix[:-1]

file_map = {}
for benchmark in benchmarks:    
    for pool_size in pool_sizes:
        fname = "%s-%s-pool%d.txt" % (prefix, benchmark, pool_size)
        file_map[(benchmark, pool_size)] = fname


result = {}
for k,v in file_map.items():
    fin = open(v, "r")
    # find the line starts with Final-PR@1:  ...
    for line in fin.readlines():
        if line.startswith("Final-PR@1:"):
            result[k] = float(line.split(":")[1].strip())
            break
    fin.close()

print("Pool size,", end="")
for benchmark in benchmarks:
    print("%s," % benchmark, end="")
print()

for pool_size in pool_sizes:
    print("%d" % pool_size, end="")
    for benchmark in benchmarks:
        print(",%.3f" % result[(benchmark, pool_size)], end="")
    print()
