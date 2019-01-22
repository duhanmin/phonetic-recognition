# phonetic-recognition项目介绍
采用cnn方案进行训练，与人脸识别face-recognition项目的网络结构类似<br />
phonetic-recognition项目将一个维度的音频向量处理成了一个音频特征矩阵<br />
MFCC相关知识参见“梅尔频率倒谱系数 MFCC"一文<br />
最后用cnn训练，后面过程与人脸识别类似<br />
<br />
# 音频训练与识别<br />
语音MFCCs特征处理：<br />
![image](https://github.com/duhanmin/phonetic-recognition/blob/master/images/3.png)<br /><br />
训练效果：<br />
![image](https://github.com/duhanmin/phonetic-recognition/blob/master/images/1.jpg)<br /><br />
识别效果：<br />
其中1的置信度低的原因是：我用自己的声音训练，用别人的声音去识别，虽然低，但也识别对了<br />
![image](https://github.com/duhanmin/phonetic-recognition/blob/master/images/2.jpg)<br /><br />

# 赞助
<img src="https://github.com/duhanmin/mathematics-statistics-python/blob/master/images/90f9a871536d5910cad6c10f0297fc7.jpg" width="250">