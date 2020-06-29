#!/bin/bash

echo -ne "BUILDING KNN..."
cd ./knn-template
make clean
output=$(make 2>&1)
if [[ $output != *"make:"* ]]; then
    echo -e "\rBUILDING SUCCESSFUL."
else
    echo -e "\rBUILD ERROR :"
    echo $output
fi

# Run tests for KNN
for query in 100 1000 10000; do
    for ref in 10000 100000 1000000; do
        for n_class in 10 20; do
            for k in 10 20; do
                echo -n "$k $n_class $ref $query"
                ./knn-exec $k $n_class $ref $query
            done
        done
    done
done

cd ..

# echo -ne "BUILDING BFS..."
# cd ./bfs-template
# make clean
# output=$(make 2>&1)
# if [[ $output != *"make:"* ]]; then
#     echo -e "\rBUILDING SUCCESSFUL."
# else
#     echo -e "\rBUILD ERROR :"
#     echo $output
# fi

# # Run tests for BFS
# for nVertices in 100000 1000000 10000000; do
#     for nEdges in 1000000 10000000 100000000; do
#         ./bfs-exec 0 $nVertices $nEdges
#     done
# done;
