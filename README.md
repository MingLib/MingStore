# MingStore
A personal code Lib using for training model.
## Collection Dataset
- **Available Datasets**
  - **Training**: [VGGFace2](https://arxiv.org/pdf/1710.08092.pdf), [WebFace](https://arxiv.org/pdf/1411.7923v1.pdf), [MS1M](https://arxiv.org/pdf/1607.08221.pdf), [Glint360K](https://arxiv.org/abs/2203.15565), [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
  - **Validation**: [LFW](http://vis-www.cs.umass.edu/lfw/), [AgeDB-30](https://ibug.doc.ic.ac.uk/media/uploads/documents/agedb.pdf), [CA-LFW](https://arxiv.org/pdf/1708.08197.pdf), [CP-LFW](http://www.whdeng.cn/CPLFW/Cross-Pose-LFW.pdf)
  - **Testing**: [IJB-B](https://openaccess.thecvf.com/content_cvpr_2017_workshops/w6/papers/Whitelam_IARPA_Janus_Benchmark-B_CVPR_2017_paper.pdf), [IJB-C](http://biometrics.cse.msu.edu/Publications/Face/Mazeetal_IARPAJanusBenchmarkCFaceDatasetAndProtocol_ICB2018.pdf)
## Collection Model
for now, there are only some outstanding pretrained weights for face recognize task collected from other repos. 
### Face recongize
1. [Deepinsight/insightface](https://github.com/deepinsight/insightface), this repo supports arcface method in Pytorch, the implementation can be found [here](https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch). It has ten pretrained model weights in `.pt` file and many training logs to follow. Besides, the repo also has [model zoo](https://github.com/deepinsight/insightface/tree/master/model_zoo), which has a lot of pretrained model weights in `.onnx` file. 
   - input size (112, 112, 3)     
   - iamge should aligned by standerd five facial landmark points.
   - normalize with 0.5

|architecture|loss|dataset|
|-------|-------|-------|
|resnet18|arcface|ms1mv3|
|resnet34|arcface|ms1mv3|
|resnet50|arcface|ms1mv3|
|resnet100|arcface|ms1mv3|
|r2060|arcface|ms1mv3|
|resnet18|cosface|glink360k|
|resnet34|cosface|glink360k|
|resnet50|cosface|glink360k|
|resnet100|cosface|glink360k|

2. [TreB1eN/InsightFace_Pytorch](https://github.com/TreB1eN/InsightFace_Pytorch), this repo has two pretrained model weights and no training logs to follow. Its implementation can't be use directly.
   - input size (112, 112, 3) 
   - iamge should aligned by standerd five facial landmark points.
   - normalize with 0.5


|architecture|loss|dataset|
|-------|-------|-------|
|IR-SE50|arcface|vggface2|
|Mobilefacenet| |vggface2|

3. [timesler/facenet-pytorch](https://github.com/timesler/facenet-pytorch), this repo has two pretrained model weights. They are trained by [davidsandberg/facenet](https://github.com/davidsandberg/facenet) using softmax loss under Tensorflow frame.
   - input size (160, 160, 3) 
   - iamge should aligned by MTCNN.
   - normalize with 0.5

|architecture|loss|dataset|
|-------|-------|-------|
|inceptionresnetv1|softmax loss|vggface2|
|inceptionresnetv1|softmax loss|CASIA-Webface|
4. [cmusatyalab/openface](https://github.com/cmusatyalab/openface), Unknow but probably has pretrained model weights.
5. [ydwen/opensphere](https://github.com/ydwen/opensphere), Unknow but has a lot of pretrained model weights.

|architecture|loss|dataset|
|-------|-------|-------|
|SFNet-20 (w/o BN)|sphereface|vggface2|
|SFNet-64 (w/ BN)|sphereface|MS1M|
|IResNet-100|sphereface|MS1M|
1. [MuggleWang/CosFace_pytorch](https://github.com/MuggleWang/CosFace_pytorch) has pretrained model weights.
2. [Xiaoccer/MobileFaceNet_Pytorch](https://github.com/Xiaoccer/MobileFaceNet_Pytorch)
3. [KaiyangZhou/pytorch-center-loss](https://github.com/KaiyangZhou/pytorch-center-loss)
4.  [tengshaofeng/ResidualAttentionNetwork-pytoch](https://github.com/tengshaofeng/ResidualAttentionNetwork-pytorch)
5.   [wujiyang/Face_Pytorch](https://github.com/wujiyang/Face_Pytorch)
