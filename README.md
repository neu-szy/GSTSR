此项目fork自[basicsr](https://github.com/xinntao/BasicSR) ，详细文档见[Readme](https://github.com/xinntao/BasicSR/blob/master/README.md)  
本项目加入了GSTSR的模型文件、训练配置文件与测试配置文件  
修改相应的yml文件后，训练GSTSR：
```angular2html
python basicsr/train.py --opt options/train/GSTSR/train_GSTSR_x{your_factor}.yml
```
修改相应的yml文件后，测试GSTSR：
```angular2html
python basicsr/test.py --opt options/test/GSTSR/test_GSTSR_x{your_factor}.yml
```
预训练模型及结果见[百度网盘：7y1q](https://pan.baidu.com/s/1o4397TclOAFnxg34WSacWQ)

