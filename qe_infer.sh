export CUDA_VISIBLE_DEVICES=7

metrics=pearson

datadir=./data/qe
modeldir=./saved_qe_model
inferdir=${datadir}/infer

mkdir -p ${inferdir}

data=dev2018

python qe_model.py \
    --out_dir=${modeldir} \
    --ckpt=${modeldir}/avg_best_${metrics} \
    --inference_src_file=${datadir}/${data}.lower.de \
    --inference_mt_file=${datadir}/${data}.lower.en \
    --inference_fea_file=${datadir}/${data}.sent.tfrecord \
    --inference_output_file=${inferdir}/${data}.infer

