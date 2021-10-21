

# remove previous results
rm -r train
rm -r freeze
rm -r deploy



#Create directory
mkdir train
mkdir freeze
mkdir deploy


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

mv -r test_images deploy/.
cp  app_mt.py deploy/.

echo "######## Finished ########"
