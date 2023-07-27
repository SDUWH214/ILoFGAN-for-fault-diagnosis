### data prepare
ImgUtils.img2npy()

### train
`python train.py  --conf configs/sdu.yaml --output_dir results/sdu`

### sample
`python sample.py --gpu 0 --name results/sdu --ckpt gen_00060000.pt`

