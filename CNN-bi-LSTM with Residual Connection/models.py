import torch
import torch.nn as nn


class Resample_Channels(nn.Module):
  def __init__(self, in_channels, out_channels):
    super(Resample_Channels, self).__init__()
    self.layer = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1),
                nn.BatchNorm2d(out_channels)
            )
  def forward(self, x):
    return self.layer(x)

#a single residual block with 3 cnn layers.
class SEED_DEEP_LAYER(nn.Module):
    def __init__(self, in_channels, out_channels, in_d_0=0, in_d_1=0, stride = 1, k=4, do_pool = False, dropout = 0.01, debug = False):
        super(SEED_DEEP_LAYER, self).__init__()
        self.do_pool = do_pool
        self.debug = debug
        self.conv1 = nn.Sequential(
                        nn.Conv2d(in_channels, 100, kernel_size = k, stride = stride, padding = 'same'),
                        nn.BatchNorm2d(100),
                        nn.LeakyReLU())


        conv2_layers = [
                        nn.Conv2d(100, out_channels, kernel_size = k, stride = stride, padding = 'same'),
                        nn.BatchNorm2d(out_channels),
                        nn.LeakyReLU()
                      ]
        self.mp = nn.MaxPool2d((1,1))
        self.dropout = nn.Dropout2d(dropout)
        self.downsample = nn.AvgPool2d((1,1))
        #a way to build layers out of a list
        self.conv2 = nn.Sequential(*conv2_layers)

        conv3_layers = [
            nn.Conv2d(out_channels, out_channels, kernel_size=k, stride=stride, padding='same'),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        ]

        self.conv3 = nn.Sequential(*conv3_layers)


        self.resample_channels = False
        if in_channels != out_channels:
          self.resampler = Resample_Channels(in_channels, out_channels)
          self.resample_channels = True
        self.a = nn.LeakyReLU()


    def forward(self, x):
        residual = x
        o = self.conv1(x)
        o = self.dropout(o)
        o = self.conv2(o)
        o = self.conv3(o)

        if self.resample_channels:
          residual = self.resampler(residual)
        if self.do_pool:
          o = self.mp(o)
          o = o + self.downsample(residual)
        else:
          o = o + residual
        o = self.a(o)

        return o


#the model for CNN-bi_LSTM-RES with and without grid.
class SEED_DEEP(nn.Module):
  def __init__(self, do_pool = True, in_channels=1, is_grid = False, grid_size = (200, 62), out_channels=200, num_classes = 3, num_res_layers = 5,  ll1=1024, ll2 = 256, dropoutc=0.01, dropout1= 0.5, dropout2 = 0.5, debug = False):
    super(SEED_DEEP, self).__init__()
    self.is_grid = is_grid
    self.debug = debug
    #must use modulelist vs regular list to keep everythign in GPU and gradients flowing
    self.res_layers = nn.ModuleList()
    c = in_channels
    for r in range(num_res_layers):
      self.res_layers.append(SEED_DEEP_LAYER(in_channels = c, out_channels=out_channels, do_pool = do_pool, dropout = dropoutc, debug = debug))
      c = out_channels
    self.lstm1 = nn.LSTM(ll1, ll1, batch_first=True, bidirectional=True)
    self.lin1 = nn.Linear(ll1, ll2)
    self.lin1_5 = nn.Linear(ll2, ll2)
    self.lin2 = nn.Linear(ll2, num_classes)
    self.do1 = nn.Dropout(dropout1)
    self.do2 = nn.Dropout(dropout2)
    self.lin1_lstm = nn.Linear(ll1 * 2, ll2)
    self.lin_res = nn.Linear(ll1, ll1 * 2)
    
    self.ldown = None
    if do_pool:
      self.ldown = nn.Linear(40000, ll1)
    else:
      self.ldown = nn.Linear(out_channels * grid_size[0]* grid_size[1], ll1)
    self.la = nn.ReLU()




  def forward(self, x):

    if not self.is_grid:
      x = torch.permute(x, (0,2,1))
      x = x.unsqueeze(1)

    o = x
    for i in range(len(self.res_layers)):
      o = self.res_layers[i](o)
    o = o.view(o.shape[0], -1)
    o = self.ldown(o)
    res = o
    res = self.lin_res(res)
    
    o, _ = self.lstm1(o)
    o = o + res
    o = self.do1(o)
    o = self.la(self.lin1_lstm(o))
    o = self.do2(o)
    o = self.lin2(o)
    return o