for dataset in "moocen"; do
  cmd="python build_graph.py -dataset $dataset -no_course_dependency -no_user_act"
  echo $cmd
  $cmd
  for model in "GCN"; do
    cmd="python main.py -dataset $dataset -model $model -output_model"
    echo $cmd
    $cmd
    cmd="python postprocess.py -dataset $dataset -model $model"
    echo $cmd
    $cmd
  done
done

for dataset in "mooczh"; do
  cmd="python build_graph.py -dataset $dataset"
  echo $cmd
  $cmd
  for model in "GCN"; do
    cmd="python main.py -dataset $dataset -model $model -output_model"
    echo $cmd
    $cmd
    cmd="python postprocess.py -dataset $dataset -model $model"
    echo $cmd
    $cmd
  done
done
