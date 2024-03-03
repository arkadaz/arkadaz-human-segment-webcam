Green screen? <br />
Nahhh. <br />
Just use AI. <br />

This is real-time human segmentation. <br />
Model_pretrain: MobileOne_S4:Imagenet <br />
Training dataset: Supervisely Person Dataset <br />
Quantize: quantize static setting 
{per_channel=True, reduce_range=True,"ActivationSymmetric":True, "WeightSymmetric":True} <br />
 <br />
 <br />
Step0 <br />
Please install obs, cuda, and cudnn. <br />
Step1 <br />
pip install -r requirements.txt <br />
Step2 <br />
python main.py <br />
