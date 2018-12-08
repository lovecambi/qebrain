export CUDA_VISIBLE_DEVICES=7

metrics=pearson

datadir=./data/qe
modeldir=./saved_qe_model
inferdir=./data/qe/infer

mkdir -p ${inferdir}

data=dev

python qe_model.py \
    --out_dir=${modeldir} \
    --ckpt=${modeldir}/avg_best_${metrics} \
    --inference_src_file=${datadir}/${data}.lower.src \
    --inference_mt_file=${datadir}/${data}.lower.mt \
    --inference_fea_file=${datadir}/${data}.std.tfrecord \
    --inference_output_file=${inferdir}/${data}.infer

