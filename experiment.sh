if [ ! -d "tmp/" ]; then
  mkdir tmp/
fi
if [ ! -d "result/" ]; then
  mkdir result/
fi
for epochs in 2 5 10 20 35 60 100 150; do
  for dataset in "moocen" "mooczh"; do
    for model in "TextCNN" "LSTM" "GCN" "GCN_LSTM" "MLP"; do
      for t in {1..10}; do
        cmd="python main.py -dataset $dataset -model $model -epochs $epochs"
        echo $cmd
        $cmd
      done
    done
  done
done
