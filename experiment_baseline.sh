for dataset in "moocen" "mooczh"; do
  for model in "LSTM" "TextCNN"; do
    for feature_dim in 36; do
      for seed in {0..4}; do
        cmd="python main.py -dataset $dataset -model $model -feature_dim $feature_dim -output base.txt -seed $seed"
        echo $cmd
        $cmd
      done
      mv result/base.txt result/${model}_${dataset}_dim${feature_dim}.txt
    done
  done
done
