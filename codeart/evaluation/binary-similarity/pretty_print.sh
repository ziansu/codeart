#!/bin/bash


if [ $# -ne 1 ]; then
    echo "Usage: $0 <pattern_prefix>"
    exit 1
fi

FIN_PATTERN_PREFIX=$1

echo "PR@1 for pool size 32, 50, 100, 200, 300, 500"
cat $FIN_PATTERN_PREFIX*32.txt|egrep "Final-PR@1"|sed 's/.*:\s*\(\S*\)/\1/g'
cat $FIN_PATTERN_PREFIX*50.txt|egrep "Final-PR@1"|sed 's/.*:\s*\(\S*\)/\1/g'
cat $FIN_PATTERN_PREFIX*100.txt|egrep "Final-PR@1"|sed 's/.*:\s*\(\S*\)/\1/g'
cat $FIN_PATTERN_PREFIX*200.txt|egrep "Final-PR@1"|sed 's/.*:\s*\(\S*\)/\1/g'
cat $FIN_PATTERN_PREFIX*300.txt|egrep "Final-PR@1"|sed 's/.*:\s*\(\S*\)/\1/g'
cat $FIN_PATTERN_PREFIX*500.txt|egrep "Final-PR@1"|sed 's/.*:\s*\(\S*\)/\1/g'

