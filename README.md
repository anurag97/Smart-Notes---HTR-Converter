## Smart Notes ##
Handwritten Notes converter from image of handwritten text to text file built using Neural network and Computer vision for image segmentation
<br>
<br>Project is mainly divided in  parts:
<br>
<br>1.Image preprocessing
<br>2.Line segmentation
<br>3.Word segmetation
<br>4.Model training and validation based on words images from IAM dataset
<br>5.Recognizing words using pretrained Model and Combining output to output.txt


<br><b>Image preprocessing:</b><br>
Image processing like resizing image, smoothing, binarization is done in this step.
![example](./doc/input.jpg)<br>
![example](./doc/out1.jpg)

<br>

<br><b>Line Segmentation:</b><br>
In this step the entire image is segmented into line such that one image corresponds to one line.<br>
![example](./doc/line.jpeg)

<br>

<br><b>Word Segmentation:</b><br>
In this one by one each line is segmented into words such that one image corresponds to one word.<br>
![example](./doc/words.png)

<br>

<br><b>Model Training and Validation:</b><br>
<br><b>1.IAM dataset</b>:<br>
It consist of large number of handwritten words images in two files : words.tgz(actual images), words.txt(ground truth) using this dataset as input model is trained.
<br>
<br><b>2.Model Configuration</b>:<br>
It consists of 5 CNN layers, 2 RNN (LSTM) layers and the CTC loss and decoding layer. The illustration below gives an overview of the NN (green: operations, pink: data flowing through NN) and here follows a short description:
The input image is a gray-value image and has a size of 128x32
5 CNN layers map the input image to a feature sequence of size 32x256
2 LSTM layers with 256 units propagate information through the sequence and map the sequence to a matrix of size 32x80. Each matrix-element represents a score for one of the 80 characters at one of the 32 time-steps
The CTC layer either calculates the loss value given the matrix and the ground-truth text (when training), or it decodes the matrix to the final text with best path decoding or beam search decoding (when inferring)
Batch size is set to 50<br>

![example](./doc/nn_overview.png)

<br><b>Recognizing words using pretrained Model and Combining output to output.txt</b><br>
In this step words are recognized using pretrained model and appended to output.txt line by line.
