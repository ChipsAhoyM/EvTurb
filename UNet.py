from arch import *
from Blocks import Event_Guided_Deformable_Conv
from flow import IFNet

class DeblurNet(nn.Module):
    def __init__(self):
        super(DeblurNet, self).__init__()
    # conv-img:
        self.conv_1 = nn.Sequential(
            Conv2D(3, 64, 3),
            Dense_block(64, 16)
        )
        self.conv_2 = nn.Sequential(
            Conv2D(128, 128, 2, 2, padding=0),
            Dense_block(128, 32)
        )
        self.conv_3 = nn.Sequential(
            Conv2D(256, 256, 2, 2, padding=0),
            Dense_block(256, 64)
        )
        self.conv_1e = nn.Sequential(
            Conv2D(32, 64, 3),
            Dense_block(64, 16)
        )
        self.conv_2e = nn.Sequential(
            Conv2D(128, 128, 2, 2, padding=0),
            Dense_block(128, 32)
        )
        self.conv_3e = nn.Sequential(
            Conv2D(256, 256, 2, 2, padding=0),
            Dense_block(256, 64)
        )

        self.fusion = nn.Sequential(
            Conv2D(512*2, 512, 1, padding=0),
            ResidualBlock(512)
        )
    # deconv 
        self.deconv_2 = DeConv2D(512, 256)

        self.conv_5 = nn.Sequential(
            Conv2D(256*2, 128, 1, padding=0),
            Dense_block(128, 32)
        )

        self.deconv_1 = DeConv2D(256, 128)

        self.conv_6 = nn.Sequential(
            Conv2D(128*2, 32, 1, padding=0),
            Dense_block(32, 8)
        )

    # prediction
        self.predConv = nn.Sequential(
            nn.Conv2d(64, 3, 5, padding=2),
        )
        
        self.skip_conv1 = Event_Guided_Deformable_Conv(256)
        self.skip_conv2 = Event_Guided_Deformable_Conv(128)
        
        self.tanh = nn.Tanh()

    def forward(self, image, event):
        c1 = self.conv_1(image)
        c2 = self.conv_2(c1)
        c3 = self.conv_3(c2)
        
        ce0 = self.conv_1e(event)
        ce1 = self.conv_2e(ce0)
        ce2 = self.conv_3e(ce1)

        m3 = torch.cat([c3, ce2], dim=1)
        fusion = self.fusion(m3)
        
        c2 = self.skip_conv1(c2, ce1)
        
        dc2 = self.deconv_2(fusion)  
        m2 = torch.cat([c2, dc2], dim=1)
        c5 = self.conv_5(m2)
        
        c1 = self.skip_conv2(c1, ce0)
        dc1 = self.deconv_1(c5)
        m1 = torch.cat([c1, dc1], dim=1)
        c6 = self.conv_6(m1)

        pred = self.tanh(self.predConv(c6)) + image

        return pred
    

class Whole(nn.Module):
    def __init__(self):
        super(Whole, self).__init__()
        self.deblur = DeblurNet()
        self.flow = IFNet()
    
    def forward(self, image, event, var, batch_idx = None):
        pred = self.deblur(image, event)
        pred = self.flow(pred, var, batchidx = batch_idx)
        return pred
    