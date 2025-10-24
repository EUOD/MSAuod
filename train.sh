### 训练命令
nohup xxx > logs/xxx.log 2>&1 & tail -f logs/xxx.log
###### 示例
CUDA_VISIBLE_DEVICES=0 nohup python train.py -c configs/yaml/dfine_hgnetv2_s_mg.yml --seed=0 > train.log 2>&1 & tail -f train.log  
##### 普通训练命令
CUDA_VISIBLE_DEVICES=0 python train.py -c configs/yaml/deim_dfine_hgnetv2_s_mg.yml --seed=0  
### 测试命令
python train.py -c configs/yaml/dfine_hgnetv2_s_mg.yml --test-only -r outputs/dfine_duo2/best_stg1.pth

### 推理命令(字体和框的大小请看tools/inference/torch_inf.py的draw函数注释)
python tools/inference/torch_inf.py -c configs/yaml/deim_dfine_hgnetv2_s_mg.yml -r outputs/deim-S+LAWDS/best_stg1.pth --input dataset/datasets/test/images --output inference_results/exp -t 0.4

### 计算yml的参数量和计算量功能
python tools/benchmark/get_info.py -c configs/yaml/deim_dfine_hgnetv2_s_mg.yml

### 输出yml的全部参数  
python show_yml_param.py -c configs/yaml/dfine_hgnetv2_s_mg.yml

### COCO格式数据集信息输出脚本(输出类别数和类别id、输出每个类别的实例数量)
python dataset/coco_analyzer.py dataset/datasets/train/annotations/train.json

CUDA_VISIBLE_DEVICES=0 nohup python train.py --seed=0 -r outputs/deim_hgnetv2_s-MSCB-Spatial_Frequency_Attention/last.pth > train.log 2>&1 & tail -f train.log

echo "🛑 清理所有训练进程..." && pkill -f "python train.py" && sleep 5 && pkill -9 -f "python train.py" 2>/dev/null && rm -f train.pid && echo "✅ 清理完成" && ps aux | grep "python train.py" | grep -v grep || echo "✅ 确认：无训练进程运行"