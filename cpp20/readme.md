# 1. conda install -c conda-forge osqp-eigen （https://github.com/robotology/osqp-eigen）
# 2. apt-get install clang-format
# 3. find ./cpp20 -path ./cpp20/googletest -prune -o -iname '*.h' -o -iname '*.cpp' -o -iname '*.cc' -exec clang-format -i {} +