if [ "$#" -ne 1 ]; then
    echo "Provide one command line argument"
    exit
fi

export PROJECT="cloze-325807"
export BUCKET="seq_trans_cloze"
export REGION="us-west1"
export TFVERSION='2.3'

JOBNAME=$1

MODELDIR=gs://${BUCKET}/training_output/${JOBNAME}

gcloud ai-platform jobs submit training $JOBNAME \
  --region=$REGION \
  --module-name=source.main \
  --package-path=./source \
  --job-dir=$MODELDIR \
  --staging-bucket=gs://$BUCKET \
  --config=./gcp_training_config.yaml \
  --runtime-version=$TFVERSION \
  --packages clickstream_transformer-0.0.1-py3-none-any.whl \
  -- \
  --input_data=gs://${BUCKET}/data/amazon_beauty_bert4rec_trunc_50 \
  --model_dir=${MODELDIR} \
  --num_encoder_layers=1 \
  --num_attention_heads=2 \
#  --init_ckpt_dir=gs://seq_trans_cloze/training_output/test_26_lr_5e4_batch_1024/ckpts
