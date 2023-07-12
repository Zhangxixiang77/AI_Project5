import torch
from torch import nn
from model.ImgOnly import ImgOnly
from model.TextOnly import TextOnly


class MultiModal(nn.Module):
    def __init__(self, args):
        super(MultiModal, self).__init__()
        self.TextModule_ = TextOnly(args)
        self.ImgModule_ = ImgOnly(args)

        self.multihead_attn = nn.MultiheadAttention(embed_dim=1000, num_heads=2, batch_first=True)
        self.linear_text_k1 = nn.Linear(1000, 1000)
        self.linear_text_v1 = nn.Linear(1000, 1000)
        self.linear_img_k2 = nn.Linear(1000, 1000)
        self.linear_img_v2 = nn.Linear(1000, 1000)

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=1000, nhead=2, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=2)

        self.classifier_img = nn.Sequential(
            nn.Linear(1000, 500),
            nn.ReLU(),
            nn.Linear(500, 200),
            nn.ReLU(),
            nn.Linear(200, 40),
            nn.ReLU(),
            nn.Linear(40, 3),
        )
        self.classifier_text = nn.Sequential(
            nn.Linear(1000, 500),
            nn.ReLU(),
            nn.Linear(500, 200),
            nn.ReLU(),
            nn.Linear(200, 40),
            nn.ReLU(),
            nn.Linear(40, 3),
        )
        self.classifier_multi = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2000, 1000),
            nn.ReLU(),
            nn.Dropout(p=args.dropout),
            nn.Linear(1000, 200),
            nn.ReLU(),
            nn.Linear(200, 40),
            nn.ReLU(),
            nn.Linear(40, 3),
        )

    def forward(self, bach_text=None, bach_img=None):
        if bach_text is not None and bach_img is None:
            _, text_out = self.TextModule_(bach_text)
            text_out = self.classifier_text(text_out)
            return text_out, None, None

        if bach_text is None and bach_img is not None:
            img_out = self.ImgModule_(bach_img)
            img_out = self.classifier_img(img_out)
            return None, img_out, None

        _, text_out = self.TextModule_(bach_text)  
        img_out = self.ImgModule_(bach_img) 

        multi_out = torch.cat((text_out, img_out), 1)

        #multi_out = self.Multihead_self_attention(text_out, img_out)

        #multi_out = self.Transformer_Encoder(text_out, img_out)

        text_out = self.classifier_text(text_out)
        img_out = self.classifier_img(img_out)
        multi_out = self.classifier_multi(multi_out)
        return text_out, img_out, multi_out

    def Multihead_self_attention(self, text_out, img_out):
       
        text_k1 = self.linear_text_k1(text_out)
        text_v1 = self.linear_text_v1(text_out)
        img_k2 = self.linear_img_k2(img_out)
        img_v2 = self.linear_img_v2(img_out)
       
        key = torch.stack((text_k1, img_k2), dim=1)  
        value = torch.stack((text_v1, img_v2), dim=1)  
        query = torch.stack((text_out, img_out), dim=1)  
        
        attn_output, attn_output_weights = self.multihead_attn(query, key, value)
        return attn_output

    def Transformer_Encoder(self, text_out, img_out):
        multimodal_sequence = torch.stack((text_out, img_out), dim=1)  
        return self.transformer_encoder(multimodal_sequence)