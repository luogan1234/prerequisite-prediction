for dataset in "moocen" "mooczh"; do
  for model in "TextCNN" "LSTM" "MLP"; do
  for feature_dim in 6 12 18 24 30; do
    for t in {1..4}; do
      cmd="python main.py -dataset $dataset -model $model -feature_dim $feature_dim -output base.txt"
      echo $cmd
      $cmd
    done
    mv result/base.txt result/${model}_${dataset}_dim${feature_dim}.txt
  done
done
