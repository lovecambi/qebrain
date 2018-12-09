export CUDA_VISIBLE_DEVICES=7
datadir=./data/qe
vocabdir=./data/vocab
exp_modeldir=./saved_exp_model

modeldir=./saved_qe_model
mkdir -p $modeldir

#rm -rf $modeldir/*

python qe_model.py \
      --src=lower.de \
      --mt=lower.en \
      --fea=sent.tfrecord \
      --lab=lower.en.hter \
      --train_prefix=${datadir}/train2018 \
      --dev_prefix=${datadir}/dev2018 \
      --test_prefix=${datadir}/test2017 \
      --vocab_prefix=${vocabdir}/vocab120k \
      --max_vocab_size=120000 \
      --out_dir=${modeldir} \
      --optimizer=lazyadam \
      --warmup_steps=8000 \
      --learning_rate=2.0 \
      --num_train_steps=75000 \
      --steps_per_stats=100 \
      --steps_per_external_eval=1000 \
      --rnn_units=128 \
      --rnn_layers=1 \
      --embedding_size=512 \
      --num_units=512 \
      --num_layers=2 \
      --ffn_inner_dim=512 \
      --qe_batch_size=64 \
      --infer_batch_size=64 \
      --metrics=pearson \
      --use_hf=False \
      --num_buckets=4 \
      --dim_hf=17 \
      --train_level=sent \
      --avg_ckpts=True \
      --fixed_exp=True \
      --label_smoothing=0.1 \
      --exp_model_dir=${exp_modeldir}

