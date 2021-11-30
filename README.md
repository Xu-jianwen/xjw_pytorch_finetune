# xjw_pytorch_finetune
微调预训练分类模型的pytorch实现

环境配置：

    conda create -n pytorch_finetune python=3.9
    conda activate pytorch_finetune
    pip install -r requirements.txt
- - -
标签格式：

Crude_Oil_Tanker/Crude_Oil_Tanker_084.bmp,0

提供了split_dataset.py用于划分训练集和测试集
- - -
开始训练：

    python finetune.py --cuda_id 0 --data_root 你存放数据集的目录 --dataset 数据集 --model_name --batch_size --num_epochs
- - -
训练完成后可使用eval.py评价模型的性能

可使用heatmap.py绘制热力图查看网络激活部位
