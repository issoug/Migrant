dataset=FIREFOX

python -u main.py \
  --dataset ${dataset} \
  --model Migrant \
  --device cuda:0 \
  --seq_len 5000 \
  --train_epochs 50 \
  --batch_size 64 \
  --learning_rate 1e-4 \
  --optimizer Adam \
  --eval_metrics Accuracy Precision Recall F1-score \
  --save_metric F1-score \
  --save_name firefox  \
  --ratio 100


python -u test.py \
  --dataset ${dataset} \
  --model Migrant \
  --device cuda:0 \
  --seq_len 5000 \
  --batch_size 64 \
  --eval_metrics Accuracy Precision Recall F1-score \
  --load_name firefox