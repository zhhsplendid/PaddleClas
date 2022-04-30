#!/bin/bas

set -xe

batch_size=${1:-"128"}
device_id=${2:-"3"}
use_cinn=${3:-"True"}
speed_optimization=${4:-"True"}
logname=${5:-"./log/default"}

### 指定单卡
export CUDA_VISIBLE_DEVICES=${device_id}

### 存储分配策略
#FLAGS_allocator_strategy="naive_best_fit"

### VLOG
#export GLOG_v=12
export GLOG_vmodule=instruction=20,graph_compiler=20,compiler=20
#export GLOG_logtostderr=1

### cinn 相关参数
if [ ${use_cinn} == "True" ]; then
  export FLAGS_use_cinn="True"
  export FLAGS_allow_cinn_ops="batch_norm;batch_norm_grad;conv2d;conv2d_grad;elementwise_add;elementwise_add_grad;relu;relu_grad;sum"
  export FLAGS_cinn_use_new_fusion_pass="True"
  #export FLAGS_allow_cinn_ops="conv2d;conv2d_grad;elementwise_add;elementwise_add_grad;relu;relu_grad;sum"
fi

### paddle 侧性能优化参数设置
use_dali="False"
enable_addto="False"
fuse_elewise_add_act_ops="False"
if [ ${speed_optimization} == "True" ]; then
  fuse_elewise_add_act_ops="True"
  enable_addto="True"
  #export FLAGS_allocator_strategy="naive_best_fit"
  #export FLAGS_fraction_of_gpu_memory_to_use=1.0
  export FLAGS_fraction_of_gpu_memory_to_use=0.8
  export FLAGS_max_inplace_grad_add=8
  export FLAGS_cudnn_exhaustive_search=1
  # 以下参数 V100上不设置，A100开启
  #export FLAGS_conv_workspace_size_limit=4000 #MB
  #export NVIDIA_TF32_OVERRIDE=1
  #use_dali="True"
fi

### 打印参数值
config=ppcls/configs/ImageNet/ResNet/ResNet50.yaml
echo "================ Running Configurations ================"
echo "CUDA_VISIBLE_DEVICES                     : $CUDA_VISIBLE_DEVICES"
echo "batch_size                               : ${batch_size}"
echo "use_dali                     : ${use_dali}"
echo "enable_addto                 : ${enable_addto}"
echo "fuse_elewise_add_act_ops             : ${fuse_elewise_add_act_ops}"
echo "FLAGS_fraction_of_gpu_memory_to_use      : ${FLAGS_fraction_of_gpu_memory_to_use}"
echo "FLAGS_max_inplace_grad_add           : ${FLAGS_max_inplace_grad_add}"
echo "FLAGS_cudnn_exhaustive_search            : ${FLAGS_cudnn_exhaustive_search}"
echo "FLAGS_conv_workspace_size_limit          : ${FLAGS_conv_workspace_size_limit}"
echo "NVIDIA_TF32_OVERRIDE             : ${NVIDIA_TF32_OVERRIDE}"
echo "FLAGS_use_cinn                   : ${FLAGS_use_cinn}"
echo "FLAGS_allow_cinn_ops             : ${FLAGS_allow_cinn_ops}"
echo ""

### 启动训练
python3.7 -u ppcls/static/train.py \
         -c ${config} \
         -o use_gpu=True \
         -o print_interval=10 \
         -o is_distributed=False \
         -o enable_addto=${enable_addto} \
         -o fuse_elewise_add_act_ops=${fuse_elewise_add_act_ops} \
         -o epochs=1 \
         -o DataLoader.Train.sampler.batch_size=${batch_size} \
         -o DataLoader.Train.dataset.image_root=./dataset/ILSVRC2012 \
         -o DataLoader.Train.dataset.cls_label_path=./dataset/ILSVRC2012/train_list.txt \
         -o DataLoader.Train.loader.num_workers=8 \
         -o Global.save_interval=10000 \
         -o Global.eval_interval=10000 \
         -o Global.eval_during_train=False \
         -o Global.use_dali=${use_dali} > "${logname}_train.log" 2>&1 &
### 启动显存使用监控
nvidia-smi --id=$CUDA_VISIBLE_DEVICES --query-compute-apps=used_memory --format=csv -lms 100 > "${logname}_mem_usage.log" 2>&1 &


