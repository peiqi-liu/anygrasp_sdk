# mamba create -n anygrasp python=3.7 -y
# mamba activate anygrasp
mamba install -c "nvidia/label/cuda-11.7.0" cuda-toolkit -y
pip install torch torchvision ninja
pip install git+https://github.com/pccws/MinkowskiEngine --no-deps
pip install Pillow 
pip install scipy tqdm 
pip install graspnetAPI 
pip install open3d
pip install numpy==1.21.6
cd pointnet2
python setup.py install
cd ..

cp grasp_detection/gsnet_versions/gsnet.cpython-37m-x86_64-linux-gnu.so grasp_detection/gsnet.so
cp license_registration/lib_cxx_versions/lib_cxx.cpython-37m-x86_64-linux-gnu.so grasp_detection/lib_cxx.so