#!/usr/bin/env bash

# New VM
rm -rf sample_data yolov3
git clone https://github.com/ultralytics/yolov3
sudo apt-get install zip
#git clone https://github.com/NVIDIA/apex && cd apex && pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" . --user && cd .. && rm -rf apex
sudo conda install -yc conda-forge scikit-image pycocotools
# python3 -c "from yolov3.utils.google_utils import gdrive_download; gdrive_download('193Zp_ye-3qXMonR1nZj3YyxMtQkMy50k','coco2014.zip')"
python3 -c "from yolov3.utils.google_utils import gdrive_download; gdrive_download('1WQT6SOktSe8Uw6r10-2JhbEhMY5DJaph','coco2014.zip')"
sudo shutdown

# Re-clone
rm -rf yolov3  # Warning: remove existing
git clone https://github.com/ultralytics/yolov3 # master
bash yolov3/data/get_coco2017.sh
# git clone -b test --depth 1 https://github.com/ultralytics/yolov3 test  # branch
cd yolov3
python3 test.py --weights ultralytics68.pt --task benchmark

# Mount local SSD
lsblk
sudo mkfs.ext4 -F /dev/nvme0n1
sudo mkdir -p /mnt/disks/nvme0n1
sudo mount /dev/nvme0n1 /mnt/disks/nvme0n1
sudo chmod a+w /mnt/disks/nvme0n1
cp -r coco /mnt/disks/nvme0n1

# Train
python3 train.py

# Resume
python3 train.py --resume

# Detect
python3 detect.py

# Test
python3 test.py --save-json

# Kill All
t=ultralytics/yolov3:v206
docker kill $(docker ps -a -q --filter ancestor=$t)
t=ultralytics/yolov3:v208
docker kill $(docker ps -a -q --filter ancestor=$t)

# Evolve
sudo -s
t=ultralytics/yolov3:v206
docker kill $(docker ps -a -q --filter ancestor=$t)
for i in 4 5 6 7
do
  docker pull $t && docker run --gpus all -d --ipc=host -v "$(pwd)"/data:/usr/src/data $t bash utils/evolve.sh $i
  # docker pull $t && docker run --gpus all -d --ipc=host -v "$(pwd)"/out:/usr/src/out $t bash utils/evolve.sh $i
  # docker pull $t && nvidia-docker run -d -v "$(pwd)"/coco:/usr/src/coco $t bash utils/evolve.sh $i
  # docker pull $t && nvidia-docker run -d -v /mnt/disks/nvme0n1/coco:/usr/src/coco $t bash utils/evolve.sh $i
  sleep 180
done

# Evolve
sudo -s
t=ultralytics/yolov3:v208
docker kill $(docker ps -a -q --filter ancestor=$t)
for i in 0 1
do
  # docker pull $t && docker run --gpus all -d --ipc=host -v "$(pwd)"/data:/usr/src/data $t bash utils/evolve.sh $i
  docker pull $t && docker run --gpus all -d --ipc=host -v "$(pwd)"/out:/usr/src/out $t bash utils/evolve.sh $i
  # docker pull $t && nvidia-docker run -d -v "$(pwd)"/coco:/usr/src/coco $t bash utils/evolve.sh $i
  # docker pull $t && nvidia-docker run -d -v /mnt/disks/nvme0n1/coco:/usr/src/coco $t bash utils/evolve.sh $i
  sleep 180
done

# Git pull
git pull https://github.com/ultralytics/yolov3  # master
git pull https://github.com/ultralytics/yolov3 test  # branch

# Test Darknet training
python3 test.py --weights ../darknet/backup/yolov3.backup

# Copy last.pt TO bucket
gsutil cp yolov3/weights/last1gpu.pt gs://ultralytics

# Copy last.pt FROM bucket
gsutil cp gs://ultralytics/last.pt yolov3/weights/last.pt
wget https://storage.googleapis.com/ultralytics/yolov3/last_v1_0.pt -O weights/last_v1_0.pt
wget https://storage.googleapis.com/ultralytics/yolov3/best_v1_0.pt -O weights/best_v1_0.pt

# Reproduce tutorials
rm results*.txt  # WARNING: removes existing results
python3 train.py --nosave --data data/coco_1img.data && mv results.txt results0r_1img.txt
python3 train.py --nosave --data data/coco_10img.data && mv results.txt results0r_10img.txt
python3 train.py --nosave --data data/coco_100img.data && mv results.txt results0r_100img.txt
# python3 train.py --nosave --data data/coco_100img.data --transfer && mv results.txt results3_100imgTL.txt
python3 -c "from utils import utils; utils.plot_results()"
# gsutil cp results*.txt gs://ultralytics
gsutil cp results.png gs://ultralytics
sudo shutdown

# Reproduce mAP
python3 test.py --save-json --img 608
python3 test.py --save-json --img 416
python3 test.py --save-json --img 320
sudo shutdown

# Benchmark script
git clone https://github.com/ultralytics/yolov3  # clone our repo
git clone https://github.com/NVIDIA/apex && cd apex && pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" . --user && cd .. && rm -rf apex  # install nvidia apex
python3 -c "from yolov3.utils.google_utils import gdrive_download; gdrive_download('1HaXkef9z6y5l4vUnCYgdmEAj61c6bfWO','coco.zip')"  # download coco dataset (20GB)
cd yolov3 && clear && python3 train.py --epochs 1  # run benchmark (~30 min)

# Unit tests
python3 detect.py  # detect 2 persons, 1 tie
python3 test.py --data data/coco_32img.data  # test mAP = 0.8
python3 train.py --data data/coco_32img.data --epochs 5 --nosave  # train 5 epochs
python3 train.py --data data/coco_1cls.data --epochs 5 --nosave  # train 5 epochs
python3 train.py --data data/coco_1img.data --epochs 5 --nosave  # train 5 epochs

# AlexyAB Darknet
gsutil cp -r gs://sm6/supermarket2 .  # dataset from bucket
rm -rf darknet && git clone https://github.com/AlexeyAB/darknet && cd darknet && wget -c https://pjreddie.com/media/files/darknet53.conv.74  # sudo apt install libopencv-dev && make
./darknet detector calc_anchors data/coco_img64.data -num_of_clusters 9 -width 320 -height 320  # kmeans anchor calculation
./darknet detector train ../supermarket2/supermarket2.data ../yolo_v3_spp_pan_scale.cfg darknet53.conv.74 -map -dont_show # train spp
./darknet detector train ../yolov3/data/coco.data ../yolov3-spp.cfg darknet53.conv.74 -map -dont_show # train spp coco

#Docker
sudo docker kill "$(sudo docker ps -q)"
sudo docker pull ultralytics/yolov3:v0
sudo nvidia-docker run -it -v "$(pwd)"/coco:/usr/src/coco ultralytics/yolov3:v0


t=ultralytics/yolov3:v70 && sudo docker pull $t && sudo nvidia-docker run -it -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --epochs 273 --batch 16 --accum 4 --pre --bucket yolov4 --name 70 --device 0 --multi
t=ultralytics/yolov3:v0 && sudo docker pull $t && sudo nvidia-docker run -it -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --epochs 273 --batch 16 --accum 4 --pre --bucket yolov4 --name 71 --device 0 --multi --img-weights

t=ultralytics/yolov3:v73 && sudo docker pull $t && sudo nvidia-docker run -it -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --epochs 27 --batch 16 --accum 4 --pre --bucket yolov4 --name 73 --device 5 --cfg cfg/yolov3s.cfg
t=ultralytics/yolov3:v74 && sudo docker pull $t && sudo nvidia-docker run -it -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --epochs 27 --batch 16 --accum 4 --pre --bucket yolov4 --name 74 --device 0 --cfg cfg/yolov3s.cfg
t=ultralytics/yolov3:v75 && sudo docker pull $t && sudo nvidia-docker run -it -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --epochs 27 --batch 16 --accum 4 --pre --bucket yolov4 --name 75 --device 7 --cfg cfg/yolov3s.cfg
t=ultralytics/yolov3:v76 && sudo docker pull $t && sudo nvidia-docker run -it -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --epochs 27 --batch 16 --accum 4 --pre --bucket yolov4 --name 76 --device 0 --cfg cfg/yolov3-spp.cfg

t=ultralytics/yolov3:v79 && sudo docker pull $t && sudo nvidia-docker run -it -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --epochs 27 --batch 16 --accum 4 --pre --bucket yolov4 --name 79 --device 5
t=ultralytics/yolov3:v80 && sudo docker pull $t && sudo nvidia-docker run -it -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --epochs 27 --batch 16 --accum 4 --pre --bucket yolov4 --name 80 --device 0
t=ultralytics/yolov3:v81 && sudo docker pull $t && sudo nvidia-docker run -it -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --epochs 27 --batch 16 --accum 4 --pre --bucket yolov4 --name 81 --device 7
t=ultralytics/yolov3:v82 && sudo docker pull $t && sudo nvidia-docker run -it -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --epochs 27 --batch 16 --accum 4 --pre --bucket yolov4 --name 82 --device 0 --cfg cfg/yolov3s.cfg

t=ultralytics/yolov3:v83 && sudo docker pull $t && sudo nvidia-docker run -it -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --epochs 273 --batch 16 --accum 4 --pre --bucket yolov4 --name 83 --device 6 --multi --nosave
t=ultralytics/yolov3:v84 && sudo docker pull $t && sudo nvidia-docker run -it -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --epochs 273 --batch 16 --accum 4 --pre --bucket yolov4 --name 84 --device 0 --multi
t=ultralytics/yolov3:v85 && sudo docker pull $t && sudo nvidia-docker run -it -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --epochs 273 --batch 16 --accum 4 --pre --bucket yolov4 --name 85 --device 0 --multi
t=ultralytics/yolov3:v86 && sudo docker pull $t && sudo nvidia-docker run -it -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --epochs 273 --batch 16 --accum 4 --pre --bucket yolov4 --name 86 --device 1 --multi
t=ultralytics/yolov3:v87 && sudo docker pull $t && sudo nvidia-docker run -it -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --epochs 273 --batch 16 --accum 4 --pre --bucket yolov4 --name 87 --device 2 --multi
t=ultralytics/yolov3:v88 && sudo docker pull $t && sudo nvidia-docker run -it -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --epochs 273 --batch 16 --accum 4 --pre --bucket yolov4 --name 88 --device 3 --multi
t=ultralytics/yolov3:v89 && sudo docker pull $t && sudo nvidia-docker run -it -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --epochs 27 --batch 16 --accum 4 --pre --bucket yolov4 --name 89 --device 1
t=ultralytics/yolov3:v90 && sudo docker pull $t && sudo nvidia-docker run -it -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --epochs 27 --batch 16 --accum 4 --pre --bucket yolov4 --name 90 --device 0 --cfg cfg/yolov3-spp-matrix.cfg
t=ultralytics/yolov3:v91 && sudo docker pull $t && sudo nvidia-docker run -it -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --epochs 27 --batch 16 --accum 4 --pre --bucket yolov4 --name 91 --device 0 --cfg cfg/yolov3-spp-matrix.cfg

t=ultralytics/yolov3:v92 && sudo docker pull $t && sudo nvidia-docker run -it -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --epochs 27 --batch 16 --accum 4 --pre --bucket yolov4 --name 92 --device 0
t=ultralytics/yolov3:v93 && sudo docker pull $t && sudo nvidia-docker run -it -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --epochs 27 --batch 16 --accum 4 --pre --bucket yolov4 --name 93 --device 0 --cfg cfg/yolov3-spp-matrix.cfg


#SM4
t=ultralytics/yolov3:v96 && sudo docker pull $t && sudo nvidia-docker run -it -v "$(pwd)"/data:/usr/src/data $t python3 train.py --weights 'ultralytics68.pt' --epochs 1000 --img 320 --batch 32 --accum 2 --pre --bucket yolov4 --name 96 --device 0 --multi --cfg cfg/yolov3-spp-3cls.cfg --data ../data/sm4/out.data --nosave
t=ultralytics/yolov3:v97 && sudo docker pull $t && sudo nvidia-docker run -it -v "$(pwd)"/data:/usr/src/data $t python3 train.py --weights 'ultralytics68.pt' --epochs 1000 --img 320 --batch 32 --accum 2 --pre --bucket yolov4 --name 97 --device 4 --multi --cfg cfg/yolov3-spp-3cls.cfg --data ../data/sm4/out.data --nosave
t=ultralytics/yolov3:v98 && sudo docker pull $t && sudo nvidia-docker run -it -v "$(pwd)"/data:/usr/src/data $t python3 train.py --weights 'ultralytics68.pt' --epochs 1000 --img 320 --batch 16 --accum 4 --pre --bucket yolov4 --name 98 --device 5 --multi --cfg cfg/yolov3-spp-3cls.cfg --data ../data/sm4/out.data --nosave
t=ultralytics/yolov3:v113 && sudo docker pull $t && sudo nvidia-docker run -it -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --batch 16 --accum 4 --pre --bucket yolov4 --name 101 --device 7 --multi --nosave

t=ultralytics/yolov3:v102 && sudo docker pull $t && sudo nvidia-docker run -it -v "$(pwd)"/data:/usr/src/data $t python3 train.py --weights 'yolov3-tiny.pt' --epochs 1000 --img 320 --batch 64 --accum 1 --pre --bucket yolov4 --name 102 --device 0 --cfg cfg/yolov3-tiny-3cls.cfg --data ../data/sm4/out.data --nosave --cache
t=ultralytics/yolov3:v103 && sudo docker pull $t && sudo nvidia-docker run -it -v "$(pwd)"/data:/usr/src/data $t python3 train.py --weights 'yolov3-tiny.pt' --epochs 500 --img 320 --batch 64 --accum 1 --pre --bucket yolov4 --name 103 --device 0 --cfg cfg/yolov3-tiny-3cls.cfg --data ../data/sm4/out.data --nosave --cache
t=ultralytics/yolov3:v104 && sudo docker pull $t && sudo nvidia-docker run -it -v "$(pwd)"/data:/usr/src/data $t python3 train.py --weights 'yolov3-tiny.pt' --epochs 500 --img 320 --batch 64 --accum 1 --pre --bucket yolov4 --name 104 --device 0 --cfg cfg/yolov3-tiny-3cls.cfg --data ../data/sm4/out.data --nosave --cache
t=ultralytics/yolov3:v105 && sudo docker pull $t && sudo nvidia-docker run -it -v "$(pwd)"/data:/usr/src/data $t python3 train.py --weights 'yolov3-tiny.pt' --epochs 500 --img 320 --batch 64 --accum 1 --pre --bucket yolov4 --name 105 --device 0 --cfg cfg/yolov3-tiny-3cls.cfg --data ../data/sm4/out.data --nosave --cache
t=ultralytics/yolov3:v106 && sudo docker pull $t && sudo nvidia-docker run -it -v "$(pwd)"/data:/usr/src/data $t python3 train.py --weights 'yolov3-tiny.pt' --epochs 500 --img 320 --batch 64 --accum 1 --pre --bucket yolov4 --name 106 --device 0 --cfg cfg/yolov3-tiny-3cls-sm4.cfg --data ../data/sm4/out.data --nosave --cache
t=ultralytics/yolov3:v107 && sudo docker pull $t && sudo nvidia-docker run -it -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --batch 32 --accum 2 --epochs 27 --pre --bucket yolov4 --name 107 --device 5 --nosave --cfg cfg/yolov3-spp3.cfg
t=ultralytics/yolov3:v108 && sudo docker pull $t && sudo nvidia-docker run -it -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --batch 32 --accum 2 --epochs 27 --pre --bucket yolov4 --name 108 --device 7 --nosave

t=ultralytics/yolov3:v109 && sudo docker pull $t && sudo nvidia-docker run -it -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --epochs 273 --batch 16 --accum 4 --pre --bucket yolov4 --name 109 --device 4 --multi --nosave
t=ultralytics/yolov3:v110 && sudo docker pull $t && sudo nvidia-docker run -it -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --epochs 273 --batch 16 --accum 4 --pre --bucket yolov4 --name 110 --device 3 --multi --nosave

t=ultralytics/yolov3:v83 && sudo docker pull $t && sudo nvidia-docker run -it -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --batch 32 --accum 2 --epochs 27 --pre --bucket yolov4 --name 111 --device 0
t=ultralytics/yolov3:v112 && sudo docker pull $t && sudo nvidia-docker run -it -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --batch 32 --accum 2 --epochs 27 --pre --bucket yolov4 --name 112 --device 1 --nosave
t=ultralytics/yolov3:v113 && sudo docker pull $t && sudo nvidia-docker run -it -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --batch 32 --accum 2 --epochs 27 --pre --bucket yolov4 --name 113 --device 2 --nosave
t=ultralytics/yolov3:v114 && sudo docker pull $t && sudo nvidia-docker run -it -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --batch 32 --accum 2 --epochs 27 --pre --bucket yolov4 --name 114 --device 2 --nosave
t=ultralytics/yolov3:v113 && sudo docker pull $t && sudo nvidia-docker run -it -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --batch 32 --accum 2 --epochs 27 --pre --bucket yolov4 --name 115 --device 5 --nosave  --cfg cfg/yolov3-spp3.cfg
t=ultralytics/yolov3:v116 && sudo docker pull $t && sudo nvidia-docker run -it -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --batch 32 --accum 2 --epochs 27 --pre --bucket yolov4 --name 116 --device 1 --nosave

t=ultralytics/yolov3:v83 && sudo docker pull $t && sudo nvidia-docker run -it -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --batch 16 --accum 4 --epochs 27 --pre --bucket yolov4 --name 117 --device 0 --nosave --multi
t=ultralytics/yolov3:v118 && sudo docker pull $t && sudo nvidia-docker run -it -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --batch 16 --accum 4 --epochs 27 --pre --bucket yolov4 --name 118 --device 5 --nosave --multi
t=ultralytics/yolov3:v119 && sudo docker pull $t && sudo nvidia-docker run -it -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --batch 32 --accum 2 --epochs 27 --pre --bucket yolov4 --name 119 --device 1 --nosave
t=ultralytics/yolov3:v120 && sudo docker pull $t && sudo nvidia-docker run -it -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --batch 32 --accum 2 --epochs 27 --pre --bucket yolov4 --name 120 --device 2 --nosave
t=ultralytics/yolov3:v121 && sudo docker pull $t && sudo nvidia-docker run -it -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --batch 32 --accum 2 --epochs 27 --pre --bucket yolov4 --name 121 --device 0 --nosave --cfg cfg/csresnext50-panet-spp.cfg
t=ultralytics/yolov3:v122 && sudo docker pull $t && sudo nvidia-docker run -it -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --batch 32 --accum 2 --epochs 273 --pre --bucket yolov4 --name 122 --device 2 --nosave
t=ultralytics/yolov3:v123 && sudo docker pull $t && sudo nvidia-docker run -it -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --batch 32 --accum 2 --epochs 273 --pre --bucket yolov4 --name 123 --device 5 --nosave

t=ultralytics/yolov3:v124 && sudo docker pull $t && sudo nvidia-docker run -it -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --batch 64 --accum 1 --epochs 27 --pre --bucket yolov4 --name 124 --device 0 --nosave --cfg yolov3-tiny
t=ultralytics/yolov3:v124 && sudo docker pull $t && sudo nvidia-docker run -d -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --batch 64 --accum 1 --epochs 27 --pre --bucket yolov4 --name 125 --device 1 --nosave --cfg yolov3-tiny2
t=ultralytics/yolov3:v124 && sudo docker pull $t && sudo nvidia-docker run -it -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --batch 64 --accum 1 --epochs 27 --pre --bucket yolov4 --name 126 --device 1 --nosave --cfg yolov3-tiny3
t=ultralytics/yolov3:v127 && sudo docker pull $t && sudo nvidia-docker run -it -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --batch 64 --accum 1 --epochs 27 --pre --bucket yolov4 --name 127 --device 0 --nosave --cfg yolov3-tiny4
t=ultralytics/yolov3:v124 && sudo docker pull $t && sudo nvidia-docker run -d -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --batch 64 --accum 1 --epochs 273 --pre --bucket yolov4 --name 128 --device 1 --nosave --cfg yolov3-tiny2 --multi
t=ultralytics/yolov3:v129 && sudo docker pull $t && sudo nvidia-docker run -it -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --batch 64 --accum 1 --epochs 273 --pre --bucket yolov4 --name 129 --device 0 --nosave --cfg yolov3-tiny2

t=ultralytics/yolov3:v130 && sudo docker pull $t && sudo nvidia-docker run -it -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --batch 32 --accum 2 --epochs 27 --pre --bucket yolov4 --name 130 --device 0 --nosave
t=ultralytics/yolov3:v133 && sudo docker pull $t && sudo nvidia-docker run -it -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --batch 22 --accum 3 --epochs 250 --pre --bucket yolov4 --name 131 --device 0 --nosave --multi
t=ultralytics/yolov3:v130 && sudo docker pull $t && sudo nvidia-docker run -it -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --batch 32 --accum 2 --epochs 27 --pre --bucket yolov4 --name 132 --device 0 --nosave --data coco2014.data
t=ultralytics/yolov3:v133 && sudo docker pull $t && sudo nvidia-docker run -it -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --batch 22 --accum 3 --epochs 27 --pre --bucket yolov4 --name 133 --device 0 --nosave --multi
t=ultralytics/yolov3:v134 && sudo docker pull $t && sudo nvidia-docker run -it -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --batch 32 --accum 2 --epochs 27 --pre --bucket yolov4 --name 134 --device 0 --nosave --data coco2014.data

t=ultralytics/yolov3:v135 && sudo docker pull $t && sudo nvidia-docker run -it -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --batch 24 --accum 3 --epochs 270 --pre --bucket yolov4 --name 135 --device 0 --nosave --multi --data coco2014.data
t=ultralytics/yolov3:v136 && sudo docker pull $t && sudo nvidia-docker run -it -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --batch 24 --accum 3 --epochs 270 --pre --bucket yolov4 --name 136 --device 0 --nosave --multi --data coco2014.data

t=ultralytics/yolov3:v137 && sudo docker pull $t && sudo nvidia-docker run -it -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --batch 32 --accum 2 --epochs 27 --pre --bucket yolov4 --name 137 --device 7 --nosave --data coco2014.data
t=ultralytics/yolov3:v137 && sudo docker pull $t && sudo nvidia-docker run -it -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --batch 32 --accum 2 --epochs 27 --bucket yolov4 --name 138 --device 6 --nosave --data coco2014.data

t=ultralytics/yolov3:v140 && sudo docker pull $t && sudo nvidia-docker run -it -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --batch 32 --accum 2 --epochs 27 --pre --bucket yolov4 --name 140 --device 1 --nosave --data coco2014.data --arc uBCE
t=ultralytics/yolov3:v141 && sudo docker pull $t && sudo nvidia-docker run -it -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --batch 32 --accum 2 --epochs 27 --pre --bucket yolov4 --name 141 --device 0 --nosave --data coco2014.data --arc uBCE
t=ultralytics/yolov3:v142 && sudo docker pull $t && sudo nvidia-docker run -it -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --batch 32 --accum 2 --epochs 27 --pre --bucket yolov4 --name 142 --device 1 --nosave --data coco2014.data --arc uBCE

t=ultralytics/yolov3:v146 && sudo docker pull $t && sudo nvidia-docker run -d -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --img 320 --batch 64 --accum 1 --epochs 27 --pre --bucket yolov4 --name 146 --device 0 --nosave --data coco2014.data
t=ultralytics/yolov3:v147 && sudo docker pull $t && sudo nvidia-docker run -it -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --img 320 --batch 64 --accum 1 --epochs 27 --pre --bucket yolov4 --name 147 --device 1 --nosave --data coco2014.data
t=ultralytics/yolov3:v148 && sudo docker pull $t && sudo nvidia-docker run -it -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --img 320 --batch 64 --accum 1 --epochs 27 --pre --bucket yolov4 --name 148 --device 2 --nosave --data coco2014.data
t=ultralytics/yolov3:v149 && sudo docker pull $t && sudo nvidia-docker run -it -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --img 320 --batch 64 --accum 1 --epochs 27 --pre --bucket yolov4 --name 149 --device 3 --nosave --data coco2014.data
t=ultralytics/yolov3:v150 && sudo docker pull $t && sudo nvidia-docker run -it -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --img 320 --batch 64 --accum 1 --epochs 27 --pre --bucket yolov4 --name 150 --device 4 --nosave --data coco2014.data
t=ultralytics/yolov3:v151 && sudo docker pull $t && sudo nvidia-docker run -d -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --img 320 --batch 64 --accum 1 --epochs 27 --pre --bucket yolov4 --name 151 --device 5 --nosave --data coco2014.data
t=ultralytics/yolov3:v152 && sudo docker pull $t && sudo nvidia-docker run -d -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --img 320 --batch 64 --accum 1 --epochs 27 --pre --bucket yolov4 --name 152 --device 6 --nosave --data coco2014.data
t=ultralytics/yolov3:v153 && sudo docker pull $t && sudo nvidia-docker run -d -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --img 320 --batch 64 --accum 1 --epochs 27 --pre --bucket yolov4 --name 153 --device 7 --nosave --data coco2014.data

t=ultralytics/yolov3:v154 && sudo docker pull $t && sudo nvidia-docker run -it -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --img 320 --batch 64 --accum 1 --epochs 27 --pre --bucket yolov4 --name 154 --device 0 --nosave --data coco2014.data
t=ultralytics/yolov3:v155 && sudo docker pull $t && sudo nvidia-docker run -it -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --img 320 --batch 64 --accum 1 --epochs 27 --pre --bucket yolov4 --name 155 --device 0 --nosave --data coco2014.data --arc defaultpw

t=ultralytics/yolov3:v156 && sudo docker pull $t && sudo nvidia-docker run -it -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --img 320 --batch 64 --accum 1 --epochs 27 --pre --bucket yolov4 --name 156 --device 5 --nosave --data coco2014.data --arc defaultpw
t=ultralytics/yolov3:v157 && sudo docker pull $t && sudo nvidia-docker run -it -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --img 320 --batch 64 --accum 1 --epochs 27 --pre --bucket yolov4 --name 157 --device 6 --nosave --data coco2014.data --arc defaultpw
t=ultralytics/yolov3:v158 && sudo docker pull $t && sudo nvidia-docker run -it -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --img 320 --batch 64 --accum 1 --epochs 27 --pre --bucket yolov4 --name 158 --device 7 --nosave --data coco2014.data --arc defaultpw

t=ultralytics/yolov3:v159 && sudo docker pull $t && sudo nvidia-docker run -it -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --img 320 --batch 64 --accum 1 --epochs 27 --pre --bucket yolov4 --name 159 --device 0 --nosave --data coco2014.data --arc defaultpw
t=ultralytics/yolov3:v160 && sudo docker pull $t && sudo nvidia-docker run -it -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --img 320 --batch 64 --accum 1 --epochs 27 --pre --bucket yolov4 --name 160 --device 1 --nosave --data coco2014.data --arc defaultpw
t=ultralytics/yolov3:v161 && sudo docker pull $t && sudo nvidia-docker run -it -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --img 320 --batch 64 --accum 1 --epochs 27 --pre --bucket yolov4 --name 161 --device 2 --nosave --data coco2014.data --arc defaultpw
t=ultralytics/yolov3:v162 && sudo docker pull $t && sudo nvidia-docker run -it -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --img 320 --batch 64 --accum 1 --epochs 27 --pre --bucket yolov4 --name 162 --device 3 --nosave --data coco2014.data --arc defaultpw
t=ultralytics/yolov3:v163 && sudo docker pull $t && sudo nvidia-docker run -it -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --img 320 --batch 64 --accum 1 --epochs 27 --pre --bucket yolov4 --name 163 --device 4 --nosave --data coco2014.data --arc defaultpw
t=ultralytics/yolov3:v164 && sudo docker pull $t && sudo nvidia-docker run -it -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --img 320 --batch 64 --accum 1 --epochs 27 --pre --bucket yolov4 --name 164 --device 5 --nosave --data coco2014.data --arc defaultpw
t=ultralytics/yolov3:v165 && sudo docker pull $t && sudo nvidia-docker run -it -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --img 320 --batch 64 --accum 1 --epochs 27 --pre --bucket yolov4 --name 165 --device 6 --nosave --data coco2014.data --arc defaultpw
t=ultralytics/yolov3:v166 && sudo docker pull $t && sudo nvidia-docker run -it -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --img 320 --batch 64 --accum 1 --epochs 27 --pre --bucket yolov4 --name 166 --device 6 --nosave --data coco2014.data --arc defaultpw
t=ultralytics/yolov3:v167 && sudo docker pull $t && sudo nvidia-docker run -it -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --img 320 --batch 64 --accum 1 --epochs 27 --pre --bucket yolov4 --name 167 --device 7 --nosave --data coco2014.data --arc defaultpw

t=ultralytics/yolov3:v168 && sudo docker pull $t && sudo nvidia-docker run -it -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --img 320 --batch 64 --accum 1 --epochs 27 --pre --bucket yolov4 --name 168 --device 5 --nosave --data coco2014.data --arc defaultpw
t=ultralytics/yolov3:v169 && sudo docker pull $t && sudo nvidia-docker run -it -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --img 320 --batch 64 --accum 1 --epochs 27 --pre --bucket yolov4 --name 169 --device 6 --nosave --data coco2014.data --arc defaultpw
t=ultralytics/yolov3:v170 && sudo docker pull $t && sudo nvidia-docker run -it -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --img 320 --batch 64 --accum 1 --epochs 27 --pre --bucket yolov4 --name 170 --device 7 --nosave --data coco2014.data --arc defaultpw
t=ultralytics/yolov3:v171 && sudo docker pull $t && sudo nvidia-docker run -it -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --img 320 --batch 64 --accum 1 --epochs 27 --pre --bucket yolov4 --name 171 --device 4 --nosave --data coco2014.data --arc defaultpw
t=ultralytics/yolov3:v172 && sudo docker pull $t && sudo nvidia-docker run -it -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --img 320 --batch 64 --accum 1 --epochs 27 --pre --bucket yolov4 --name 172 --device 3 --nosave --data coco2014.data --arc defaultpw
t=ultralytics/yolov3:v173 && sudo docker pull $t && sudo nvidia-docker run -it -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --img 320 --batch 64 --accum 1 --epochs 27 --pre --bucket yolov4 --name 173 --device 2 --nosave --data coco2014.data --arc defaultpw
t=ultralytics/yolov3:v174 && sudo docker pull $t && sudo nvidia-docker run -it -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --img 320 --batch 64 --accum 1 --epochs 27 --pre --bucket yolov4 --name 174 --device 1 --nosave --data coco2014.data --arc defaultpw
t=ultralytics/yolov3:v175 && sudo docker pull $t && sudo nvidia-docker run -it -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --img 320 --batch 64 --accum 1 --epochs 27 --pre --bucket yolov4 --name 175 --device 0 --nosave --data coco2014.data --arc defaultpw

t=ultralytics/yolov3:v177 && sudo docker pull $t && sudo nvidia-docker run -it -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --img 416 --batch 22 --accum 3 --epochs 273 --pre --bucket yolov4 --name 177 --device 0 --nosave --data coco2014.data --multi
t=ultralytics/yolov3:v178 && sudo docker pull $t && sudo nvidia-docker run -it -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --img 416 --batch 22 --accum 3 --epochs 273 --pre --bucket yolov4 --name 178 --device 0 --nosave --data coco2014.data --multi
t=ultralytics/yolov3:v179 && sudo docker pull $t && sudo nvidia-docker run -it -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --img 416 --batch 22 --accum 3 --epochs 273 --pre --bucket yolov4 --name 179 --device 0 --nosave --data coco2014.data --multi --cfg yolov3s-18a.cfg

t=ultralytics/yolov3:v143 && sudo docker build -t $t . && sudo docker push $t

t=ultralytics/yolov3:v179 && sudo docker pull $t && sudo nvidia-docker run -it -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --data coco2014.data --img-size 640 --epochs 10 --batch 22 --accum 3 --weights '' --arc defaultpw --pre --multi --bucket yolov4 --name 179
t=ultralytics/yolov3:v180 && sudo docker pull $t && sudo nvidia-docker run -it -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --data coco2014.data --img-size 640 --epochs 10 --batch 22 --accum 3 --weights '' --arc defaultpw --pre --multi --bucket yolov4 --name 180
t=ultralytics/yolov3:v183 && sudo docker pull $t && sudo nvidia-docker run -it -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --data coco2014.data --img-size 640 --epochs 10 --batch 22 --accum 3 --weights '' --arc defaultpw --pre --multi --bucket yolov4 --name 181 --cfg yolov3s9a-640.cfg
t=ultralytics/yolov3:v183 && sudo docker pull $t && sudo nvidia-docker run -it -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --data coco2014.data --img-size 640 --epochs 10 --batch 22 --accum 3 --weights '' --arc defaultpw --pre --multi --bucket yolov4 --name 182 --cfg yolov3s9a-320-640.cfg
t=ultralytics/yolov3:v183 && sudo docker pull $t && sudo nvidia-docker run -it -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --data coco2014.data --img-size 640 --epochs 10 --batch 22 --accum 3 --weights '' --arc defaultpw --pre --multi --bucket yolov4 --name 183 --cfg yolov3s15a-640.cfg
t=ultralytics/yolov3:v183 && sudo docker pull $t && sudo nvidia-docker run -it -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --data coco2014.data --img-size 640 --epochs 10 --batch 22 --accum 3 --weights '' --arc defaultpw --pre --multi --bucket yolov4 --name 184 --cfg yolov3s15a-320-640.cfg

t=ultralytics/yolov3:v185 && sudo docker pull $t && sudo nvidia-docker run -it -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --data coco2014.data --img-size 640 --epochs 10 --batch 22 --accum 3 --weights '' --arc defaultpw --pre --multi --bucket yolov4 --name 185
t=ultralytics/yolov3:v186 && sudo docker pull $t && sudo nvidia-docker run -it -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --data coco2014.data --img-size 640 --epochs 10 --batch 22 --accum 3 --weights '' --arc defaultpw --pre --multi --bucket yolov4 --name 186
n=187 && t=ultralytics/yolov3:v$n && sudo docker pull $t && sudo nvidia-docker run -it -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --data coco2014.data --img-size 640 --epochs 10 --batch 22 --accum 3 --weights '' --arc defaultpw --pre --multi --bucket yolov4 --name $n
t=ultralytics/yolov3:v189 && sudo docker pull $t && sudo nvidia-docker run -it -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --data coco2014.data --img-size 640 --epochs 10 --batch 22 --accum 3 --weights '' --arc defaultpw --pre --multi --bucket yolov4 --name 188 --cfg yolov3s15a-320-640.cfg
n=190 && t=ultralytics/yolov3:v$n && sudo docker pull $t && sudo nvidia-docker run -it -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --data coco2014.data --img-size 640 --epochs 10 --batch 22 --accum 3 --weights '' --arc defaultpw --pre --multi --bucket yolov4 --name $n
n=191 && t=ultralytics/yolov3:v$n && sudo docker pull $t && sudo nvidia-docker run -it -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --data coco2014.data --img-size 640 --epochs 10 --batch 22 --accum 3 --weights '' --arc defaultpw --pre --multi --bucket yolov4 --name $n
n=192 && t=ultralytics/yolov3:v$n && sudo docker pull $t && sudo nvidia-docker run -it -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --data coco2014.data --img-size 640 --epochs 10 --batch 22 --accum 3 --weights '' --arc defaultpw --pre --multi --bucket yolov4 --name $n

n=193 && t=ultralytics/yolov3:v$n && sudo docker pull $t && sudo nvidia-docker run -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --data coco2014.data --img-size 640 --epochs 10 --batch 22 --accum 3 --weights '' --arc defaultpw --pre --multi --bucket yolov4 --name $n
n=194 && t=ultralytics/yolov3:v$n && sudo docker pull $t && sudo nvidia-docker run -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --data coco2014.data --img-size 640 --epochs 10 --batch 22 --accum 3 --weights '' --arc defaultpw --pre --multi --bucket yolov4 --name $n
n=195 && t=ultralytics/yolov3:v$n && sudo docker pull $t && sudo nvidia-docker run -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --data coco2014.data --img-size 640 --epochs 10 --batch 22 --accum 3 --weights '' --arc defaultpw --pre --multi --bucket yolov4 --name $n
n=196 && t=ultralytics/yolov3:v$n && sudo docker pull $t && sudo nvidia-docker run -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --data coco2014.data --img-size 640 --epochs 10 --batch 22 --accum 3 --weights '' --arc defaultpw --pre --multi --bucket yolov4 --name $n

n=197 && t=ultralytics/yolov3:v$n && sudo docker pull $t && sudo nvidia-docker run -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --data coco2014.data --img-size 640 --epochs 273 --batch 22 --accum 3 --weights '' --arc defaultpw --pre --multi --bucket yolov4 --name $n
n=198 && t=ultralytics/yolov3:v$n && sudo docker pull $t && sudo nvidia-docker run -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --data coco2014.data --img-size 640 --epochs 273 --batch 22 --accum 3 --weights '' --arc defaultpw --pre --multi --bucket yolov4 --name $n

# knife
n=199 && t=ultralytics/yolov3:v$n && sudo docker pull $t && sudo nvidia-docker run -v "$(pwd)"/out:/usr/src/out $t python3 train.py --data ../out/data.data --img-size 608 --epochs 100 --batch 8 --accum 8 --weights ultralytics68.pt --arc defaultpw --pre --multi --bucket ultralytics/athena --name $n --device 0
n=200 && t=ultralytics/yolov3:v$n && sudo docker pull $t && sudo nvidia-docker run -v "$(pwd)"/out:/usr/src/out $t python3 train.py --data ../out/data.data --img-size 608 --epochs 100 --batch 8 --accum 8 --weights ultralytics68.pt --arc defaultpw --pre --multi --bucket ultralytics/athena --name $n --device 6
n=207 && t=ultralytics/yolov3:v$n && sudo docker pull $t && sudo nvidia-docker run -v "$(pwd)"/out:/usr/src/out $t python3 train.py --data ../out/data.data --img-size 608 --epochs 100 --batch 8 --accum 8 --weights ultralytics68.pt --arc defaultpw --pre --multi --bucket ultralytics/athena --name $n --device 7
n=208 && t=ultralytics/yolov3:v$n && sudo docker pull $t && sudo docker run --gpus all -v "$(pwd)"/out:/usr/src/out $t python3 train.py --data ../out/data.data --img-size 608 --epochs 10 --batch 8 --accum 8 --weights ultralytics68.pt --arc defaultpw --pre --multi --device 0
n=211 && t=ultralytics/yolov3:v$n && sudo docker pull $t && sudo docker run --gpus all -v "$(pwd)"/out:/usr/src/out $t python3 train.py --data ../out/data.data --img-size 608 --epochs 100 --batch 8 --accum 8 --weights ultralytics68.pt --arc defaultpw --pre --multi --device 0 --bucket ult/athena --name $n --cfg yolov3-spp-1cls.cfg
n=212 && t=ultralytics/yolov3:v$n && sudo docker pull $t && sudo docker run --gpus all -v "$(pwd)"/out:/usr/src/out $t python3 train.py --data ../out/data.data --img-size 608 --epochs 100 --batch 8 --accum 8 --weights ultralytics68.pt --arc defaultpw --pre --multi --device 0 --bucket ult/athena --name $n --cfg yolov3-spp-1cls.cfg
n=213 && t=ultralytics/yolov3:v$n && sudo docker pull $t && sudo docker run --gpus all -v "$(pwd)"/out:/usr/src/out $t python3 train.py --data ../out/data.data --img-size 608 --epochs 100 --batch 8 --accum 8 --weights ultralytics68.pt --arc defaultpw --pre --multi --device 0 --bucket ult/athena --name $n --cfg yolov3-spp-1cls.cfg
n=214 && t=ultralytics/yolov3:v$n && sudo docker pull $t && sudo docker run --gpus all -v "$(pwd)"/out:/usr/src/out $t python3 train.py --data ../out/data.data --img-size 608 --epochs 100 --batch 8 --accum 8 --weights ultralytics68.pt --arc defaultpw --pre --multi --device 0 --bucket ult/athena --name $n --cfg yolov3-spp-1cls.cfg
n=215 && t=ultralytics/yolov3:v$n && sudo docker pull $t && sudo docker run --gpus all -v "$(pwd)"/out:/usr/src/out $t python3 train.py --data ../out/data.data --img-size 608 --epochs 100 --batch 8 --accum 8 --weights ultralytics68.pt --arc defaultpw --pre --multi --device 0 --bucket ult/athena --name $n --cfg yolov3-spp-1cls.cfg
n=217 && t=ultralytics/yolov3:v$n && sudo docker pull $t && sudo docker run --gpus all --ipc=host -it -v "$(pwd)"/out:/usr/src/out $t python3 train.py --data ../out/data.data --img-size 608 --epochs 100 --batch 8 --accum 8 --weights ultralytics68.pt --arc default --pre --multi --device 6 --bucket ult/athena --name $n --nosave --cfg yolov3-spp-1cls.cfg
n=219 && t=ultralytics/yolov3:v215 && sudo docker pull $t && sudo docker run -d --gpus all --ipc=host -v "$(pwd)"/out:/usr/src/out $t python3 train.py --data ../out/data.data --img-size 608 --epochs 10 --batch 8 --accum 8 --weights ultralytics68.pt --arc default --pre --multi --device 0 --bucket ult/athena --name $n --nosave --cfg yolov3-spp-1cls.cfg
n=220 && t=ultralytics/yolov3:v215 && sudo docker pull $t && sudo docker run -d --gpus all --ipc=host -v "$(pwd)"/out:/usr/src/out $t python3 train.py --data ../out/data.data --img-size 608 --epochs 20 --batch 8 --accum 8 --weights ultralytics68.pt --arc default --pre --multi --device 1 --bucket ult/athena --name $n --nosave --cfg yolov3-spp-1cls.cfg
n=221 && t=ultralytics/yolov3:v215 && sudo docker pull $t && sudo docker run -d --gpus all --ipc=host -v "$(pwd)"/out:/usr/src/out $t python3 train.py --data ../out/data.data --img-size 608 --epochs 30 --batch 8 --accum 8 --weights ultralytics68.pt --arc default --pre --multi --device 2 --bucket ult/athena --name $n --nosave --cfg yolov3-spp-1cls.cfg
n=222 && t=ultralytics/yolov3:v215 && sudo docker pull $t && sudo docker run -d --gpus all --ipc=host -v "$(pwd)"/out:/usr/src/out $t python3 train.py --data ../out/data.data --img-size 608 --epochs 40 --batch 8 --accum 8 --weights ultralytics68.pt --arc default --pre --multi --device 3 --bucket ult/athena --name $n --nosave --cfg yolov3-spp-1cls.cfg
n=223 && t=ultralytics/yolov3:v215 && sudo docker pull $t && sudo docker run -d --gpus all --ipc=host -v "$(pwd)"/out:/usr/src/out $t python3 train.py --data ../out/data.data --img-size 608 --epochs 10 --batch 8 --accum 8 --weights ultralytics68.pt --arc defaultpw --pre --multi --device 0 --bucket ult/athena --name $n --nosave --cfg yolov3-spp-1cls.cfg
n=224 && t=ultralytics/yolov3:v215 && sudo docker pull $t && sudo docker run -d --gpus all --ipc=host -v "$(pwd)"/out:/usr/src/out $t python3 train.py --data ../out/data.data --img-size 608 --epochs 20 --batch 8 --accum 8 --weights ultralytics68.pt --arc defaultpw --pre --multi --device 1 --bucket ult/athena --name $n --nosave --cfg yolov3-spp-1cls.cfg
n=225 && t=ultralytics/yolov3:v215 && sudo docker pull $t && sudo docker run -d --gpus all --ipc=host -v "$(pwd)"/out:/usr/src/out $t python3 train.py --data ../out/data.data --img-size 608 --epochs 30 --batch 8 --accum 8 --weights ultralytics68.pt --arc defaultpw --pre --multi --device 0 --bucket ult/athena --name $n --nosave --cfg yolov3-spp-1cls.cfg
n=226 && t=ultralytics/yolov3:v215 && sudo docker pull $t && sudo docker run -d --gpus all --ipc=host -v "$(pwd)"/out:/usr/src/out $t python3 train.py --data ../out/data.data --img-size 608 --epochs 40 --batch 8 --accum 8 --weights ultralytics68.pt --arc defaultpw --pre --multi --device 0 --bucket ult/athena --name $n --nosave --cfg yolov3-spp-1cls.cfg
n=227 && t=ultralytics/yolov3:v$n && sudo docker pull $t && sudo docker run -it --gpus all --ipc=host -v "$(pwd)"/out:/usr/src/out $t python3 train.py --data ../out/data.data --img-size 608 --epochs 10 --batch 8 --accum 8 --weights ultralytics68.pt --arc defaultpw --multi --device 0 --bucket ult/athena --name $n --nosave --cfg yolov3-spp-1cls.cfg
n=228 && t=ultralytics/yolov3:v$n && sudo docker pull $t && sudo docker run -it --gpus all --ipc=host -v "$(pwd)"/out:/usr/src/out $t python3 train.py --data ../out/data.data --img-size 608 --epochs 20 --batch 8 --accum 8 --weights ultralytics68.pt --arc defaultpw --multi --device 0 --bucket ult/athena --name $n --nosave --cfg yolov3-spp-1cls.cfg
n=229 && t=ultralytics/yolov3:v$n && sudo docker pull $t && sudo docker run -it --gpus all --ipc=host -v "$(pwd)"/out:/usr/src/out $t python3 train.py --data ../out/data.data --img-size 608 --epochs 20 --batch 8 --accum 8 --weights ultralytics68.pt --arc defaultpw --multi --device 0 --bucket ult/athena --name $n --nosave --cfg yolov3-spp-1cls.cfg

# sm4
n=201 && t=ultralytics/yolov3:v201 && sudo docker pull $t && sudo nvidia-docker run -d -v "$(pwd)"/data:/usr/src/data $t python3 train.py --data ../data/sm4/out.data --img-size 320 --epochs 1000 --batch 64 --accum 1 --weights yolov3-tiny.pt --arc defaultpw --pre --multi --bucket ult/wer --name $n --device 0 --cfg yolov3-tiny-3cls.cfg
n=202 && t=ultralytics/yolov3:v201 && sudo docker pull $t && sudo nvidia-docker run -d -v "$(pwd)"/data:/usr/src/data $t python3 train.py --data ../data/sm4/out.data --img-size 320 --epochs 1000 --batch 64 --accum 1 --weights yolov3-tiny.pt --arc defaultpw --pre --multi --bucket ult/wer --name $n --device 1 --cfg yolov3-tiny-3cls-sm4.cfg
n=203 && t=ultralytics/yolov3:v201 && sudo docker pull $t && sudo nvidia-docker run -d -v "$(pwd)"/data:/usr/src/data $t python3 train.py --data ../data/sm4/out.data --img-size 320 --epochs 1000 --batch 64 --accum 1 --weights '' --arc defaultpw --pre --multi --bucket ult/wer --name $n --device 2 --cfg yolov3-tiny-3cls-sm4.cfg
n=204 && t=ultralytics/yolov3:v202 && sudo docker pull $t && sudo nvidia-docker run -d -v "$(pwd)"/data:/usr/src/data $t python3 train.py --data ../data/sm4/out.data --img-size 320 --epochs 1000 --batch 64 --accum 1 --weights yolov3-tiny.pt --arc defaultpw --pre --multi --bucket ult/wer --name $n --device 3 --cfg yolov3-tiny-3cls-sm4.cfg
n=205 && t=ultralytics/yolov3:v202 && sudo docker pull $t && sudo nvidia-docker run -it -v "$(pwd)"/data:/usr/src/data $t python3 train.py --data ../data/sm4/out.data --img-size 320 --epochs 1000 --batch 64 --accum 1 --weights '' --arc defaultpw --pre --multi --bucket ult/wer --name $n --device 4 --cfg yolov3-tiny-3cls-sm4.cfg
n=206 && t=ultralytics/yolov3:v$n && sudo docker pull $t && sudo docker run --gpus all -it -v "$(pwd)"/data:/usr/src/data $t python3 train.py --data ../data/sm4/out.data --img-size 320 --epochs 100 --batch 64 --accum 1 --weights yolov3-tiny.pt --arc defaultpw --pre --multi --notest --nosave --cache --device 0 --cfg yolov3-tiny-3cls.cfg
n=209 && t=ultralytics/yolov3:v$n && sudo docker pull $t && sudo docker run --gpus all -it -v "$(pwd)"/data:/usr/src/data $t python3 train.py --data ../data/sm4/out.data --img-size 320 --epochs 1000 --batch 64 --accum 1 --weights yolov3-tiny.pt --arc defaultpw --pre --multi --bucket ult/wer --name $n --nosave --cache --device 3 --cfg yolov3-tiny-3cls.cfg
n=210 && t=ultralytics/yolov3:v$n && sudo docker pull $t && sudo docker run --gpus all -it -v "$(pwd)"/data:/usr/src/data $t python3 train.py --data ../data/sm4/out.data --img-size 320 --epochs 1000 --batch 64 --accum 1 --weights yolov3-tiny.pt --arc defaultpw --pre --multi --bucket ult/wer --name $n --nosave --cache --device 1 --cfg yolov3-tiny-3cls.cfg
n=216 && t=ultralytics/yolov3:v$n && sudo docker pull $t && sudo docker run --gpus all -it -v "$(pwd)"/data:/usr/src/data $t python3 train.py --data ../data/sm4/out.data --img-size 320 --epochs 1000 --batch 64 --accum 1 --weights yolov3-tiny.pt --arc defaultpw --pre --multi --bucket ult/wer --name $n --nosave --cache --device 0 --cfg yolov3-tiny-3cls.cfg
n=218 && t=ultralytics/yolov3:v$n && sudo docker pull $t && sudo docker run --gpus all --ipc=host -it -v "$(pwd)"/data:/usr/src/data $t python3 train.py --data ../data/sm4/out.data --img-size 320 --epochs 1000 --batch 64 --accum 1 --weights yolov3-tiny.pt --arc default --pre --multi --bucket ult/wer --name $n --nosave --cache --device 7 --cfg yolov3-tiny-3cls.cfg
n=230 && t=ultralytics/athena:v$n && sudo docker pull $t && sudo docker run --gpus all --ipc=host -it -v "$(pwd)"/data:/usr/src/data $t python3 train.py --data ../data/sm4/out.data --img-size 320 --epochs 100 --batch 64 --accum 1 --weights yolov3-tiny.pt --arc defaultpw --multi --bucket ult/wer --name $n --nosave --cache --device 0 --cfg yolov3-tiny-1cls.cfg --single
n=231 && t=ultralytics/athena:v$n && sudo docker pull $t && sudo docker run --gpus all --ipc=host -it -v "$(pwd)"/data:/usr/src/data $t python3 train.py --data ../data/sm4/out.data --img-size 320 --epochs 100 --batch 64 --accum 1 --weights yolov3-tiny.pt --arc defaultpw --multi --bucket ult/wer --name $n --nosave --cache --device 1 --cfg yolov3-tiny-1cls.cfg --single
n=232 && t=ultralytics/athena:v$n && sudo docker pull $t && sudo docker run --gpus all --ipc=host -it -v "$(pwd)"/data:/usr/src/data $t python3 train.py --data ../data/sm4/out.data --img-size 320 --epochs 100 --batch 64 --accum 1 --weights yolov3-tiny.pt --arc defaultpw --multi --bucket ult/wer --name $n --nosave --cache --device 0 --cfg yolov3-tiny-1cls.cfg --single
n=233 && t=ultralytics/athena:v$n && sudo docker pull $t && sudo docker run --gpus all --ipc=host -it -v "$(pwd)"/data:/usr/src/data $t python3 train.py --data ../data/sm4/out.data --img-size 320 --epochs 100 --batch 64 --accum 1 --weights yolov3-tiny.pt --arc defaultpw --multi --bucket ult/wer --name $n --nosave --cache --device 0 --cfg yolov3-tiny-1cls.cfg --single


n=206 && t=ultralytics/yolov3:v$n && sudo docker pull $t && sudo docker run --gpus all -it --ipc=host -v "$(pwd)"/data:/usr/src/data $t python3 train.py --data ../data/sm4/out.data --img-size 320 --epochs 10 --batch 64 --accum 1 --weights yolov3-tiny.pt --arc defaultpw --pre --multi --nosave --cache --device 0 --cfg yolov3-tiny-3cls.cfg
n=206 && t=ultralytics/yolov3:v$n && sudo docker pull $t && sudo docker run --gpus all -it --ipc=host -v "$(pwd)"/data:/usr/src/data $t python3 train.py --data ../data/sm4/out.data --img-size 320 --epochs 10 --batch 64 --accum 1 --weights yolov3-tiny.pt --arc defaultpw --pre --multi --nosave --cache --device 1 --cfg yolov3-tiny-3cls.cfg

