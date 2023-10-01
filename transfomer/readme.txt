This is the readme file for the transformer architecture of the GPT 3 paper in which we only implement the Decoder only network 
Kindly refer the papers "Attention is all you need" from 2017 paper in which it is clearly explained 
The only change we did in this architecture is that we apply the softmax (Normalization) first before going for the multiheaded attention but this is reverse in the original paper 
(Kindly refer https://arxiv.org/pdf/1706.03762.pdf for more details on the transformer paper)

In the transformer_karpathy.py file there are multiple versions(4 versions) of the attention is being implemented and the clear explanation is also being given in the comments section

In the bigram.py file the whole code in different versions is presented 

The final edit i.e., the generation of the text using the transformer is given in the notebook Transfomer_decode_GPT3.ipynb so please refer to the same and doubts if any can be refered to the andrewj karpathy video/DM for more clarifications
