#!/bin/bash
PARAMS=""
while (( "$#" )); do
  case "$1" in
    -l|--lab)
      LAB=$2
      shift 2
      ;;
    --) # end argument parsing
      shift
      break
      ;;
    -*|--*=) # unsupported flags
      echo "Error: Unsupported flag $1" >&2
      exit 1
      ;;
    *) # preserve positional arguments
      PARAMS="$PARAMS $1"
      shift
      ;;
  esac
done

# set positional arguments in their proper place
eval set -- "$PARAMS"

# SHELL SORT
echo "SHELL SORT STATISTICS :"
# Sequential
echo "Sequential"
for size in 5000000, 10000000, 30000000
do
	${PWD}/$LAB/shellsort -s $size
done
# Parallel
echo "Parallel"
for size in 5000000, 10000000, 30000000
do
	for threads in 1, 2, 4, 8, 16, 32, 64
	do
		${PWD}/$LAB/shellsort-parallel -s 10 -n $threads
	done
done

exit 0

# RADIX SORT
echo "RADIX SORT STATISTICS :"
# Sequential
echo "Sequential"
for size in 5000000, 10000000, 30000000
do
	${PWD}/$LAB/shellsort -s $size
done
# Parallel
echo "Parallel"
for size in 5000000, 10000000, 30000000
do
	for threads in 1, 2, 4, 8, 16, 32, 64
	do
		${PWD}/$LAB/shellsort-parallel -s 10 -n $threads
	done
done
