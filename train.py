import argparse
import os

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as LS
import torch.utils.data as data
from torch.autograd import Variable
from tqdm import tqdm

import charset
import network

parser = argparse.ArgumentParser()
parser.add_argument(
    '--batch-size', '-N', type=int, default=128, help='batch size')
parser.add_argument(
    '--data_dir',  type=str, default='/home/danyang/mfs/data/hccr', help='folder of training images')
parser.add_argument(
    '--max-epochs', '-e', type=int, default=100, help='max epochs')
parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')
parser.add_argument(
    '--iterations', type=int, default=16, help='unroll iterations')
parser.add_argument('--checkpoint', type=int, help='unroll iterations')
parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--gpu', default='1', type=str, metavar='O',
                    help='id of gpu used')
parser.add_argument('--model_dir', default='models', help='model directory')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

train_set = charset.Charset(data_dir=args.data_dir, file_name='image_2939x200x64x64_stand.npy', single_size=16)
train_loader = data.DataLoader(
    dataset=train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
print('total images: {}; total batches: {}'.format(
    len(train_set), len(train_loader)))

model = network.FullTransfer(16, 16)
model = torch.nn.DataParallel(model).cuda()

optimizer = optim.Adam(model.parameters(), lr=args.lr)
criterion = torch.nn.L1Loss()


def resume(epoch=None):
    if epoch is None:
        print('Error')
        return
    else:
        s = 'epoch'
        model.load_state_dict(
            torch.load('checkpoint/model_{}_{:08d}.pth'.format(s, epoch))
        )


def save(epoch):
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    torch.save(model.state_dict(), os.path.join(args.model_dir, './model_{}_{:04d}.pth'.format('epoch', epoch)))


scheduler = LS.MultiStepLR(optimizer, milestones=[50, 80, 100], gamma=0.5)

last_epoch = 0
if args.checkpoint:
    resume(args.checkpoint)
    last_epoch = args.checkpoint
    scheduler.last_epoch = last_epoch - 1

for epoch in range(last_epoch + 1, args.max_epochs + 1):
    scheduler.step()
    epoch_loss = 0.0
    for idx, (data, gt) in tqdm(enumerate(train_loader)):
        style_data = Variable(data[:, 0].cuda())
        content_data = Variable(data[:, 1].cuda())
        output = model(style_data, content_data)
        gt_var = Variable(gt).cuda()
        loss = criterion(output, gt_var)
        epoch_loss += loss
        # print ('loss:', loss.data[0])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print('Epoch %d Loss is %.6f\n' % (epoch, epoch_loss / len(train_loader)))

    if epoch % 5 == 0:
        save(epoch)
