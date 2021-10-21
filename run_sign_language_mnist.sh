

# remove previous results
rm -r train
rm -r freeze
rm -r quantize
rm -r deploy
rm -r dump


#Create directory
mkdir train
mkdir keras2tf
mkdir freeze
mkdir chkpt
mkdir quantize
mkdir launchmodel
mkdir deploy
mkdir dump
mkdir deploy/images
mkdir deploy/custom_images

#Creatge 

echo "######## Training keras model and converting to TF ########"
python3 train_nn.py 

echo "######## Compiling for the DPU ########"

vai_c_tensorflow2 \
    -m ./train/quantized_model.h5 \
    -a /opt/vitis_ai/conda/envs/vitis-ai-tensorflow/arch/DPUCZDX8G/Ultra96/arch.json \
    -o ./deploy \
    -n mnist

python3 gen_images_test.py 

echo "######## Preparing deploy folder ########"

cp -r test_images deploy/.
cp  app_mt.py deploy/.

echo "######## Finished ########"
