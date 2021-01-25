if [ "$#" -ne 1 ]; then
    echo "Provide one command line argument"
    exit
fi

# Building the wheel file for our custom package
PACKAGING_DIR="packaging_area"

# it's easiest to isolate the package code in a directory before building it. It ensures other stuff don't get mixed in
rm -rf $PACKAGING_DIR
mkdir -p $PACKAGING_DIR
cp -r sequence_transformer $PACKAGING_DIR
cp setup.py $PACKAGING_DIR
cd $PACKAGING_DIR
python setup.py bdist_wheel
cp dist/*.whl ..
cd ..
rm -rf $PACKAGING_DIR
DEPENDENCIES=*.whl

# Submitting the job
PROJECT="nodal-boulder-301709"
BUCKET="seq_trans_1"
REGION="europe-west2"
RUNTIME_VERSION='2.3'

JOBNAME=$1

MODELDIR=gs://${BUCKET}/training_output/${JOBNAME}

gcloud ai-platform jobs submit training $JOBNAME \
  --region=$REGION \
  --module-name=source.trainer \
  --package-path=applications/Cloze/source \
  --packages $DEPENDENCIES \
  --job-dir=$MODELDIR \
  --staging-bucket=gs://$BUCKET \
  --config=gcp_training_config.yaml \
  --runtime-version=$RUNTIME_VERSION \
  -- \
  --input_data=gs://${BUCKET}/data/amazon_beauty_bert4rec \
  --model_dir=${MODELDIR}