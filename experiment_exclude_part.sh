for dataset in "moocen" "mooczh"; do
  cmd="python build_graph.py -dataset $dataset -no_co_occur"
  echo $cmd
  $cmd
  for t in {1..1}; do
    cmd="python main.py -dataset $dataset -model GCN"
    echo $cmd
    $cmd
  done
  mv result/GCN_${dataset}_500.txt result/GCN_${dataset}_500_no_co_occur.txt
  cmd="python build_graph.py -dataset $dataset -no_video_order"
  echo $cmd
  $cmd
  for t in {1..1}; do
    cmd="python main.py -dataset $dataset -model GCN"
    echo $cmd
    $cmd
  done
  mv result/GCN_${dataset}_500.txt result/GCN_${dataset}_500_no_video_order.txt
  cmd="python build_graph.py -dataset $dataset -no_course_dependency"
  echo $cmd
  $cmd
  for t in {1..1}; do
    cmd="python main.py -dataset $dataset -model GCN"
    echo $cmd
    $cmd
  done
  mv result/GCN_${dataset}_500.txt result/GCN_${dataset}_500_no_course_dependency.txt
  cmd="python build_graph.py -dataset $dataset -no_user_data"
  echo $cmd
  $cmd
  for t in {1..1}; do
    cmd="python main.py -dataset $dataset -model GCN"
    echo $cmd
    $cmd
  done
  mv result/GCN_${dataset}_500.txt result/GCN_${dataset}_500_no_user_data.txt
done
